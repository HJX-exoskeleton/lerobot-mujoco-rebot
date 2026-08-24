#!/usr/bin/env python3
"""Minimal physical pick test based on rebotarm_pick_place_demo.py.

No tactile geoms, no object attachment, and no direct qpos writes during the
trajectory.  The object can lift only when MuJoCo reports bilateral finger
contacts.
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import cv2


ROOT = Path(__file__).resolve().parents[0]
XML = ROOT /"sim_rebotarm_colored_grasp.xml"
ARM_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ACT_NAMES = [f"joint{i}_position" for i in range(1, 7)] + ["gripper_position"]


def ik(model, site_id, seed, target, iters=500):
    d = mujoco.MjData(model)
    q = np.asarray(seed, dtype=float).copy()
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_NAMES]
    qa = np.array([model.jnt_qposadr[i] for i in ids])
    lo = model.jnt_range[ids, 0]
    hi = model.jnt_range[ids, 1]
    for _ in range(iters):
        d.qpos[qa] = q
        mujoco.mj_forward(model, d)
        err = np.asarray(target) - d.site_xpos[site_id]
        if np.linalg.norm(err) < 0.0015:
            return q
        jp = np.zeros((3, model.nv)); jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d, jp, jr, site_id)
        J = jp[:, [model.jnt_dofadr[i] for i in ids]]
        step = J.T @ np.linalg.solve(J @ J.T + 0.035**2 * np.eye(3), err)
        n = np.linalg.norm(step)
        if n > 0.10: step *= 0.10 / n
        q = np.clip(q + step, lo, hi)
    raise RuntimeError("IK failed")


def contacts(model, data, obj_geom, left_ids, right_ids):
    lf = rf = 0.0; lc = rc = False
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if obj_geom not in pair: continue
        mujoco.mj_contactForce(model, data, i, wrench)
        f = max(float(wrench[0]), 0.0)
        if pair & left_ids: lc, lf = True, lf + f
        if pair & right_ids: rc, rf = True, rf + f
    return lc, rc, lf, rf


def tactile_from_physical_contacts(model, data, obj_geom, side):
    """Project physical finger contact force onto the non-colliding pad grid."""
    prefix = f"touch_point_{side}_"
    sites = []
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if name.startswith(prefix):
            sites.append(sid)
    sites.sort(key=lambda sid: int((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or "_000")[-3:]))
    out = np.zeros(128, dtype=np.float32)
    if len(sites) != 128:
        return out.reshape(8, 16)
    gids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_collision"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"finger_{side}_rear_collision")}
    gids.discard(-1)
    wrench = np.zeros(6)
    contact_points = []
    contact_forces = []
    for i in range(data.ncon):
        c = data.contact[i]
        if obj_geom not in (int(c.geom1), int(c.geom2)) or not ({int(c.geom1), int(c.geom2)} & gids):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        fn = max(float(wrench[0]), 0.0)
        if fn <= 0: continue
        contact_points.append(np.asarray(c.pos).copy())
        contact_forces.append(fn)
    if contact_points:
        # The front and rear collision boxes may produce two solver points,
        # but they represent one physical line on the cylindrical surface.
        p = np.average(np.asarray(contact_points), axis=0,
                       weights=np.asarray(contact_forces))
        total_fn = float(np.sum(contact_forces))
        sp = data.site_xpos[np.asarray(sites)]
        if int(model.geom_type[obj_geom]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            # Box-on-finger contact: render a triangular pressure footprint.
            centered = sp - p[None, :]
            _, _, vh = np.linalg.svd(centered - centered.mean(axis=0), full_matrices=False)
            axes = vh[:2]
            uv = centered @ axes.T
            long_axis = int(np.argmax(np.ptp(uv, axis=0)))
            short_axis = 1 - long_axis
            u = uv[:, long_axis]
            v = uv[:, short_axis]
            # One vertex points into the pad; the base widens toward the
            # contact centre, producing a small triangular highlight.
            length = 0.032
            base = 0.014
            w = ((v >= -0.004) & (v <= length) &
                 (np.abs(u) <= base * (1.0 - np.maximum(v, 0.0) / length))).astype(np.float32)
            w *= np.exp(-((u / 0.006) ** 2))
        else:
            # Cylinder-on-finger contact: spread along its axis as a line.
            d_axis = np.abs(sp[:, 2] - p[2])
            d_lateral = np.linalg.norm(sp[:, :2] - p[:2][None, :], axis=1)
            w = np.exp(-((d_lateral / 0.008) ** 2) - ((d_axis / 0.030) ** 2))
            w *= (d_lateral < 0.014)
        if w.sum() > 1e-8: out += total_fn * w / w.sum()
    grid = (out * 8.0).clip(0, 1.0).reshape(8, 16)
    # Keep one response ridge per row. This removes the parallel duplicate
    # line caused by multiple XML taxel/contact samples on the same pad.
    ridge = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        c = int(np.argmax(grid[r]))
        if grid[r, c] > 1e-5:
            ridge[r, max(0, c - 1):min(grid.shape[1], c + 2)] = grid[r, c]
    return ridge


def tactile_window(left, right, state):
    """Render two tall, center-symmetric tactile panels.

    The right panel is mirrored about the central divider so both panels use
    the same anatomical left-to-right orientation.  ``state`` is retained in
    the signature for callers, but deliberately is not drawn on the image.
    """
    def panel(a, label, mirror=False):
        a = np.nan_to_num(a).clip(0, 1)
        im = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        # Taxel arrays are 8x16; use a tall panel so the pad is displayed
        # vertically rather than as a wide strip.
        im = cv2.resize(im, (160, 320), interpolation=cv2.INTER_NEAREST)
        if mirror:
            # Mirror the complete tactile image about the future center
            # divider before adding text; the label itself must remain
            # readable and must not be mirrored.
            im = cv2.flip(im, 1)
        cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_TRIPLEX, .62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return im
    # Symmetrize only the displayed image.  The right sensor is first mapped
    # into the left sensor's coordinate frame, then the two measurements are
    # averaged.  This makes the divider an exact mirror axis while preserving
    # both sides' tactile information in the displayed intensity.
    left = np.nan_to_num(left).clip(0, 1)
    right_in_left_frame = np.fliplr(np.nan_to_num(right).clip(0, 1))
    symmetric = 0.5 * (left + right_in_left_frame)
    # Use ASCII labels because the OpenCV runtime may not have a CJK font.
    li = panel(symmetric, "left")
    ri = panel(symmetric, "right", mirror=True)
    canvas = np.hstack([li, ri])
    cv2.line(canvas, (canvas.shape[1] // 2, 0), (canvas.shape[1] // 2, canvas.shape[0]), (255,255,255), 2)
    return canvas


def manual_main():
    """Interactive diagnostic mode: never overwrite data.ctrl."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    # Manual mode keeps the load-bearing finger collisions, while taxel cells
    # are display-only and are sampled by surface distance.
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_box_collision")
    left = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("finger_left_collision", "finger_left_rear_collision")}
    right = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
             for n in ("finger_right_collision", "finger_right_rear_collision")}
    left.discard(-1); right.discard(-1)
    print("[MANUAL] 自动轨迹已关闭；请在 MuJoCo viewer 的 actuator/control 面板拖动 gripper_position。", flush=True)
    print("[MANUAL] 本模式不会写入 data.ctrl；实体接触与距离触觉图相互独立。", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cv2.namedWindow("Tactile Contact - Left | Right", cv2.WINDOW_NORMAL)
        last_display = 0.0
        last_report = 0.0
        tactile_left_ema = np.zeros((8, 16), dtype=np.float32)
        tactile_right_ema = np.zeros((8, 16), dtype=np.float32)
        tactile_left = np.zeros((8, 16), dtype=np.float32)
        tactile_right = np.zeros((8, 16), dtype=np.float32)
        try:
            while viewer.is_running():
                # Run physics in batches; GUI and OpenCV are capped at 30 Hz
                # so dragging the actuator slider remains responsive.
                for _ in range(4):
                    mujoco.mj_step(model, data)
                lc, rc, lf, rf = contacts(model, data, obj_geom, left, right)
                if data.time - last_display >= 1.0 / 30.0:
                    # Taxels are deliberately non-colliding.  Estimate the
                    # tactile image from the distance between every taxel
                    # cell and the real object surface.  Run these 256
                    # narrow-phase queries only at display rate, not every
                    # physics step, so the viewer remains responsive.
                    tactile_left = tactile_from_pad_proximity(model, data, obj_geom, "left")
                    tactile_right = tactile_from_pad_proximity(model, data, obj_geom, "right")
                    tactile_left_ema = 0.35 * tactile_left + 0.65 * tactile_left_ema
                    tactile_right_ema = 0.35 * tactile_right + 0.65 * tactile_right_ema
                    cv2.imshow("Tactile Contact - Left | Right", tactile_window(tactile_left_ema, tactile_right_ema, "MANUAL"))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                    last_display = data.time
                if data.time - last_report >= 0.25:
                    print(f"[MANUAL] contact={lc}/{rc}, force={lf:.3f}/{rf:.3f}, ncon={data.ncon}", flush=True)
                    last_report = data.time
                viewer.sync()
        finally:
            cv2.destroyAllWindows()


def configure_tactile_surface(model, data, inset):
    """Inset tactile pads and remove them from the load-bearing collision set."""
    found = False
    mujoco.mj_forward(model, data)
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or ""
        if not body.startswith(("touch_cell_left_", "touch_cell_right_", "touch_pad_left", "touch_pad_right")):
            continue
        found = True
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 0
        if inset <= 0:
            continue
        side = -1.0 if "left" in body else 1.0
        world_delta = np.array([0.0, side * inset, 0.0])
        bid = int(model.geom_bodyid[gid])
        R = data.xmat[bid].reshape(3, 3)
        model.body_pos[bid] += R.T @ world_delta
    if found:
        mujoco.mj_forward(model, data)
    return found


def enable_taxel_object_contacts(model, data, object_geom):
    """Enable only taxel↔selected-object collision (not taxel self-contact)."""
    TAXEL_BIT = 2
    model.geom_conaffinity[object_geom] |= TAXEL_BIT
    mujoco.mj_forward(model, data)
    object_pos = data.geom_xpos[object_geom].copy()
    candidates = {"left": [], "right": []}
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or ""
        if body.startswith(("touch_cell_left_", "touch_cell_right_")):
            side = "left" if "left" in body else "right"
            candidates[side].append((float(np.linalg.norm(data.geom_xpos[gid] - object_pos)), gid))
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
    # Enable every cell: each pad must be an independent contact sample.
    selected = []
    for values in candidates.values():
        selected.extend(g for _, g in sorted(values))
    for gid in selected:
        model.geom_contype[gid] = TAXEL_BIT
        # Taxel bit 2 meets ordinary object contype 1; affinity 1 prevents
        # taxel-taxel (2&1=0) while allowing object contact (1&1=1).
        model.geom_conaffinity[gid] = 1
        # A thin compliant sensing skin catches near-contact before the rigid
        # finger shell carries the load; it does not change object dynamics.
        model.geom_margin[gid] = 0.0
        model.geom_condim[gid] = 1
        model.geom_friction[gid, :] = [0.0, 0.0, 0.0]
        model.geom_solref[gid, :] = [0.01, 1.0]
    count = len(selected)
    return count


def tactile_from_taxel_contacts(model, data, obj_geom, side):
    """Read actual taxel↔object contact forces into one 8x16 grid."""
    grid = np.zeros((8, 16), dtype=np.float32)
    prefix = f"touch_cell_{side}_"
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if obj_geom not in (int(c.geom1), int(c.geom2)):
            continue
        other = int(c.geom2) if int(c.geom1) == obj_geom else int(c.geom1)
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[other])) or ""
        if not body.startswith(prefix):
            continue
        try:
            col = int(body.rsplit("_", 1)[1])
        except ValueError:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        grid[col // 16, col % 16] += max(float(wrench[0]), 0.0) * 8.0
    return np.clip(grid, 0.0, 1.0)


def tactile_from_force_sensors(model, data, side):
    """2F85-compatible direct force-sensor reader (8x16)."""
    prefix = f"touch_point_{side}_"
    sensors = []
    for sid in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or ""
        if name.startswith(prefix):
            sensors.append(sid)
    sensors.sort(key=lambda sid: int((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or "_000")[-3:]))
    if len(sensors) != 128:
        return np.zeros((8, 16), dtype=np.float32)
    values = np.zeros(128, dtype=np.float32)
    for k, sid in enumerate(sensors):
        adr = int(model.sensor_adr[sid]); dim = int(model.sensor_dim[sid])
        values[k] = min(float(np.linalg.norm(data.sensordata[adr:adr + dim])), 1.0)
    return values.reshape(16, 8).T


def tactile_from_pad_proximity(model, data, obj_geom, side):
    """Return a display-only tactile image from taxel/object surface distance.

    ``mj_geomDistance`` performs MuJoCo's narrow-phase distance query without
    creating a contact.  Thus 128 cells per side do not add solver constraints
    or impulses, while curved objects still produce a spatially correct patch.
    """
    prefix = f"touch_cell_{side}_"
    ids = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if body.startswith(prefix):
            ids.append(gid)
    ids.sort(key=lambda i: int((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                   int(model.geom_bodyid[i])) or "_000")[-3:]))
    if len(ids) != 128:
        return np.zeros((8, 16), dtype=np.float32)
    d = np.empty(128, dtype=np.float32)
    fromto = np.empty(6, dtype=np.float64)
    for k, gid in enumerate(ids):
        # 6 mm sensing reach: zero at the edge of the virtual skin and one
        # at contact/very small separation. Negative distance is penetration.
        d[k] = float(mujoco.mj_geomDistance(model, data, int(gid),
                                            int(obj_geom), 0.006, fromto))
    # Contact itself is the peak; distance away from the surface decays
    # exponentially. This produces a narrow line for a cylinder instead of
    # filling a large triangular/diagonal area with nearly equal intensity.
    sigma = 0.00030
    reach = 0.00125
    separation = np.maximum(d, 0.0)
    values = np.exp(-np.square(separation / sigma)).astype(np.float32)
    values[d > reach] = 0.0
    values[values < 0.05] = 0.0
    return values.reshape(16, 8).T.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--manual", action="store_true", help="手动拖动 viewer 控制条，不执行自动轨迹（默认）")
    ap.add_argument("--auto", action="store_true", help="显式启用自主抓取轨迹")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--object", choices=("yellow_cylinder", "red_box"),
                    default="yellow_cylinder", help="抓取目标物体")
    # 触觉 XML 已永久内缩 0.3 mm；默认不再运行时重复平移。
    ap.add_argument("--tactile-inset", type=float, default=0.0,
                    help="额外的运行时内缩距离(m)")
    args = ap.parse_args()
    if args.manual or not args.auto:
        manual_main()
        return
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    tactile_present = configure_tactile_surface(model, data, args.tactile_inset)
    print(f"[PHYS] tactile_surface={'present/non-colliding' if tactile_present else 'absent'}, additional_inset={args.tactile_inset:.4f}m")
    acts = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACT_NAMES])
    arm_ids = acts[:6]; grip_id = int(acts[6])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
    obj_name = args.object
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    obj_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_collision")
    if obj_body < 0 or obj_geom < 0:
        raise RuntimeError(f"{obj_name} model body/geom not found")
    print("[PHYS] pad collision disabled; tactile map uses per-taxel surface distance", flush=True)
    left = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_left_collision", "finger_left_rear_collision")}
    right = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_right_collision", "finger_right_rear_collision")}
    left.discard(-1); right.discard(-1)
    gmin, gmax = model.actuator_ctrlrange[grip_id]
    q0 = np.array([0.0, -0.70, -0.80, 0.0, 0.0, 0.0])
    data.qpos[:6] = q0; data.qpos[6] = gmax; data.qpos[7] = -gmax
    data.ctrl[acts] = np.r_[q0, gmax]
    mujoco.mj_forward(model, data)
    obj0 = data.xpos[obj_body].copy()
    q_above = ik(model, site, q0, obj0 + [0, 0, 0.12])
    # Keep the cylinder below the palm: grasp with the finger pads' middle
    # section instead of letting the cylinder top hit gripper_palm_collision.
    # Descend closer to the object's mid-plane before closing. This avoids
    # contacting only the upper edge with one finger.
    q_pick = ik(model, site, q_above, obj0 + [0, 0, 0.010])
    # Lift just above the table first. This shorter vertical motion is less
    # likely to unload a side when the cylinder radius has been edited.
    q_lift = ik(model, site, q_pick, obj0 + [0, 0, 0.06])
    # Reference-style physical close target; leave a small preload margin.
    # The restored optimized collision shells sit about 0.6 mm outside the
    # red-box surface at the nominal 15 mm target.  Close slightly farther so
    # MuJoCo establishes an actual bilateral contact instead of terminating
    # with only a near-distance tactile footprint.
    closed = max(float(gmin), min(float(gmax), 0.010))
    phases = [("above", q_above, gmax, 0.75), ("lower", q_pick, gmax, 0.65),
              ("close", q_pick, closed, 1.50),
              ("grip_settle", q_pick, closed, 0.40),
              # A slower lift ramp avoids unloading one side of a narrow or
              # resized cylinder; the free-space phases remain accelerated.
              ("lift", q_lift, closed, 2.20),
              ("hold", q_lift, closed, 2.00)]
    phase = 0; start = data.time; start_target = np.r_[q0, gmax]
    target = start_target.copy(); last_print = -1.0; contact_seen = False
    close_hold = None
    close_contact_dist = None
    lift_retry = 0
    viewer = None
    tactile_left_smooth = np.zeros((8, 16), dtype=np.float32)
    tactile_right_smooth = np.zeros((8, 16), dtype=np.float32)
    last_display = -1.0
    last_sync = -1.0
    main._close_retry = 0
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        cv2.namedWindow("Tactile Contact - Left | Right", cv2.WINDOW_NORMAL)
    try:
        while data.time < args.duration and phase < len(phases):
            name, aq, gt, duration = phases[phase]
            u = np.clip((data.time - start) / duration, 0, 1)
            s = u*u*(3-2*u)
            target = (1-s)*start_target + s*np.r_[aq, gt]
            if name == "close" and close_hold is not None:
                # Stop at first bilateral contact; never drive the fingers
                # farther into the cylinder just to reach a fixed qpos.
                target[6] = close_hold
            data.ctrl[acts] = target
            data.qfrc_applied[:] = 0
            mujoco.mj_forward(model, data)
            for aid in arm_ids:
                jid = int(model.actuator_trnid[aid, 0]); dof = int(model.jnt_dofadr[jid])
                data.qfrc_applied[dof] = data.qfrc_bias[dof]
            # Advance two small physics steps per Python iteration. Controls
            # remain constant across the pair, reducing Python/viewer
            # overhead without skipping contact dynamics.
            mujoco.mj_step(model, data)
            mujoco.mj_step(model, data)
            lc, rc, lf, rf = contacts(model, data, obj_geom, left, right)
            if not args.headless and data.time - last_display >= 1.0 / 30.0:
                # Taxels are display-only.  Use the same per-cell surface
                # distance map as manual mode; no taxel contacts are created.
                tl = tactile_from_pad_proximity(model, data, obj_geom, "left")
                tr = tactile_from_pad_proximity(model, data, obj_geom, "right")
                # Low-pass filter the visualization only; contact forces and
                # grasp dynamics remain untouched.
                tactile_left_smooth = 0.25 * tl + 0.75 * tactile_left_smooth
                tactile_right_smooth = 0.25 * tr + 0.75 * tactile_right_smooth
                cv2.imshow("Tactile Contact - Left | Right",
                           tactile_window(tactile_left_smooth, tactile_right_smooth, name))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[PHYS] tactile window closed by user", flush=True)
                    break
                last_display = data.time
            if name in ("lift", "hold") and (not lc or not rc):
                if name == "lift" and lift_retry < 2:
                    # A resized cylinder can lose one side during the first
                    # arm acceleration. Retry with half the remaining lift
                    # displacement instead of terminating the simulation.
                    lift_retry += 1
                    scale = 0.5 ** lift_retry
                    safe_lift = q_pick + scale * (q_lift - q_pick)
                    hold_grip = float(data.ctrl[grip_id])
                    phases[phase] = ("lift", safe_lift, hold_grip, 1.20)
                    start = data.time
                    start_target = np.r_[q_pick, hold_grip]
                    print(f"[PHYS] lift retry {lift_retry}: reduced vertical displacement", flush=True)
                    continue
                print(f"[PHYS] FAIL: bilateral contact lost during {name} ({lc}/{rc}); gripper remains closed", flush=True)
                break
            if lc and rc:
                contact_seen = True
                if name == "close" and close_hold is None:
                    close_hold = float(data.qpos[6])
                    ds = []
                    for j in range(data.ncon):
                        c = data.contact[j]; pair = {int(c.geom1), int(c.geom2)}
                        if obj_geom in pair and (pair & left or pair & right): ds.append(float(c.dist))
                    close_contact_dist = min(ds) if ds else float("nan")
                    print(f"[PHYS] contact latched at gripper={close_hold:.5f}, dist={close_contact_dist:.6f}", flush=True)
            if data.time - last_print >= 0.5:
                extras = ""
                if name == "lift":
                    pairs = []
                    for ci in range(data.ncon):
                        cc = data.contact[ci]
                        if obj_geom in (int(cc.geom1), int(cc.geom2)):
                            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(cc.geom1)) or "?"
                            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(cc.geom2)) or "?"
                            pairs.append(f"{a}:{b}")
                    extras = f", object_contacts={pairs}"
                # Compute a low-rate tactile activity statistic for the
                # terminal; visualization itself is updated above only when
                # a viewer is enabled.
                tl_report = tactile_from_pad_proximity(model, data, obj_geom, "left")
                tr_report = tactile_from_pad_proximity(model, data, obj_geom, "right")
                ta = int(np.count_nonzero(tl_report) + np.count_nonzero(tr_report))
                print(f"[PHYS] {name}: qgrip={data.qpos[6]:.4f}, contact={lc}/{rc}, taxel_active={ta}, force={lf:.3f}/{rf:.3f}, object_z={data.xpos[obj_body,2]:.4f}{extras}", flush=True)
                last_print = data.time
            if data.time - start >= duration:
                if name == "close" and not contact_seen:
                    # Object dimensions may be changed in the XML.  Give the
                    # fingers one additional, bounded closing pass instead of
                    # aborting merely because the nominal target was reached.
                    close_retry = getattr(main, "_close_retry", 0)
                    if close_retry < 1 and float(data.qpos[6]) > float(gmin) + 0.001:
                        main._close_retry = close_retry + 1
                        retry_target = max(float(gmin) + 0.001, 0.003)
                        phases[phase] = ("close", q_pick, retry_target, 0.90)
                        start = data.time
                        start_target = target.copy()
                        print(f"[PHYS] close target extended for current object size: {retry_target:.4f}", flush=True)
                        continue
                    dleft = []; dright = []; seg = np.empty(6)
                    for gid in left:
                        dleft.append(float(mujoco.mj_geomDistance(model, data, int(gid), int(obj_geom), 0.02, seg)))
                    for gid in right:
                        dright.append(float(mujoco.mj_geomDistance(model, data, int(gid), int(obj_geom), 0.02, seg)))
                    print(f"[PHYS] close diagnostics: finger_distance={min(dleft, default=float('inf')):.6f}/{min(dright, default=float('inf')):.6f}", flush=True)
                    print("[PHYS] FAIL: no bilateral physical contact; gripper remains closed", flush=True)
                    break
                if name == "lift" and data.xpos[obj_body,2] - obj0[2] < 0.03:
                    print("[PHYS] FAIL: object not retained by friction; gripper remains closed", flush=True)
                    break
                if name == "close" and close_hold is not None and phase + 1 < len(phases):
                    # Carry the actual contact opening into lift; do not
                    # switch back to the nominal closed target and deepen
                    # penetration while raising the object.
                    # A small, controlled preload keeps the object above the
                    # friction limit without driving the fingers through it.
                    # Keep a small additional preload for objects whose
                    # radius was changed in XML; otherwise the first lift
                    # acceleration can unload one side of the cylinder.
                    preload = max(float(gmin), close_hold - 0.0040)
                    phases[phase + 1] = ("grip_settle", phases[phase + 1][1], preload, phases[phase + 1][3])
                    if phase + 2 < len(phases):
                        phases[phase + 2] = ("lift", phases[phase + 2][1], preload, phases[phase + 2][3])
                    if phase + 3 < len(phases):
                        phases[phase + 3] = ("hold", phases[phase + 3][1], preload, phases[phase + 3][3])
                phase += 1; start = data.time; start_target = target.copy(); contact_seen = False
                if name != "close":
                    close_hold = None
                    close_contact_dist = None
                print(f"[PHYS] phase -> {phases[phase][0] if phase < len(phases) else 'SUCCESS'}", flush=True)
            if viewer and viewer.is_running() and data.time - last_sync >= 1.0 / 60.0:
                viewer.sync()
                last_sync = data.time
            if not args.headless:
                time.sleep(0.0005)
        # A successful pick remains interactive.  Keep the arm and gripper
        # under the final hold command until the user closes the MuJoCo
        # viewer; do not tear down the simulation immediately after SUCCESS.
        if phase == len(phases) and viewer is not None:
            print("[PHYS] SUCCESS: viewer remains open; close the MuJoCo window to exit", flush=True)
            while viewer.is_running():
                data.ctrl[acts] = target
                mujoco.mj_step(model, data)
                if data.time - last_sync >= 1.0 / 60.0:
                    viewer.sync()
                    last_sync = data.time
                if data.time - last_display >= 1.0 / 30.0:
                    tl = tactile_from_pad_proximity(model, data, obj_geom, "left")
                    tr = tactile_from_pad_proximity(model, data, obj_geom, "right")
                    tactile_left_smooth = 0.25 * tl + 0.75 * tactile_left_smooth
                    tactile_right_smooth = 0.25 * tr + 0.75 * tactile_right_smooth
                    cv2.imshow("Tactile Contact - Left | Right",
                               tactile_window(tactile_left_smooth, tactile_right_smooth, "hold"))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    last_display = data.time
                time.sleep(0.001)
    finally:
        if viewer: viewer.close()
        if not args.headless: cv2.destroyAllWindows()
    print("[PHYS] SUCCESS: bilateral contact and lift confirmed" if phase == len(phases) else "[PHYS] stopped without success")


if __name__ == "__main__":
    main()
