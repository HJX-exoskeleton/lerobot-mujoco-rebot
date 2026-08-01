"""Compare one real-policy deployment against a LeRobot demonstration set.

The comparison is phase-normalized so runs of different lengths can be
compared without pretending that their wall-clock timestamps are aligned.
Numeric Parquet columns are loaded without decoding the 1.7 GiB image payload;
only a few frames from the nearest demonstration are decoded for the visual
comparison plate.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

os.environ["MPLBACKEND"] = "Agg"
_MPL_CACHE = Path(tempfile.gettempdir()) / f"rebot_act_compare_mpl_{os.getuid()}"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE)

from rebot_act_real.policy_logging.analyze import (  # noqa: E402
    _chunk_lengths,
    _read_segment_frame,
    _style,
    load_run,
)


NUMERIC_COLUMNS = (
    "observation.state",
    "action",
    "sensor.joint_velocity",
    "sensor.imu",
    "sensor.tactile",
    "timestamp",
)


def _as_array(column) -> np.ndarray:
    return np.asarray(column.to_pylist())


def load_demonstrations(dataset_root: Path) -> list[dict[str, np.ndarray]]:
    files = sorted((Path(dataset_root) / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到示教Parquet: {dataset_root}")
    episodes: list[dict[str, np.ndarray]] = []
    for path in files:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in NUMERIC_COLUMNS if name in schema_names]
        table = pq.read_table(path, columns=columns)
        episode = {name: _as_array(table[name]) for name in columns}
        episode["_path"] = np.asarray(str(path))
        episodes.append(episode)
    return episodes


def _resample(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    flat = values.reshape(len(values), -1)
    result = np.column_stack(
        [np.interp(target, source, flat[:, index]) for index in range(flat.shape[1])]
    )
    return result.reshape((points,) + values.shape[1:])


def _phase_stack(
    episodes: list[dict[str, np.ndarray]], key: str, points: int
) -> np.ndarray | None:
    if not all(key in episode for episode in episodes):
        return None
    return np.stack([_resample(episode[key], points) for episode in episodes])


def _normalized_wasserstein(reference: np.ndarray, query: np.ndarray) -> float:
    quantiles = np.linspace(0.01, 0.99, 199)
    ref_q = np.quantile(reference, quantiles)
    query_q = np.quantile(query, quantiles)
    scale = max(float(np.std(reference)), 1e-8)
    return float(np.mean(np.abs(ref_q - query_q)) / scale)


def _derivative_rms(values: np.ndarray, time_s: np.ndarray, order: int) -> float:
    result = np.asarray(values, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    for _ in range(order):
        result = np.gradient(result, time_s, axis=0)
    return float(np.sqrt(np.mean(np.square(result))))


def _percentile_rank(reference: list[float], value: float) -> float:
    return float(100.0 * np.mean(np.asarray(reference) <= value))


def _trajectory_metrics(
    demo: np.ndarray, deployment: np.ndarray
) -> tuple[dict[str, Any], int]:
    median = np.median(demo, axis=0)
    low = np.percentile(demo, 5, axis=0)
    high = np.percentile(demo, 95, axis=0)
    phase_coverage = np.mean((deployment >= low) & (deployment <= high), axis=0)
    scale = np.std(demo.reshape(-1, demo.shape[-1]), axis=0)
    normalized_error = np.sqrt(np.mean(np.square(deployment - median), axis=0)) / np.maximum(
        scale, 1e-8
    )
    episode_rmse = np.sqrt(
        np.mean(np.square(demo - deployment[None, ...]), axis=(1, 2))
    )
    nearest = int(np.argmin(episode_rmse))
    return (
        {
            "phase_band_coverage_per_dimension": phase_coverage.tolist(),
            "phase_band_coverage_mean": float(np.mean(phase_coverage)),
            "normalized_rmse_to_demo_median_per_dimension": normalized_error.tolist(),
            "normalized_rmse_to_demo_median_mean": float(np.mean(normalized_error)),
            "nearest_episode_index": nearest,
            "nearest_episode_rmse": float(episode_rmse[nearest]),
        },
        nearest,
    )


def _distribution_metrics(
    demo: np.ndarray, deployment: np.ndarray
) -> dict[str, Any]:
    dimensions = demo.shape[-1]
    flattened = demo.reshape(-1, dimensions)
    query = deployment.reshape(-1, dimensions)
    low, high = np.percentile(flattened, [1, 99], axis=0)
    coverage = np.mean((query >= low) & (query <= high), axis=0)
    wasserstein = [
        _normalized_wasserstein(flattened[:, index], query[:, index])
        for index in range(dimensions)
    ]
    return {
        "demo_1_99_percentile_coverage_per_dimension": coverage.tolist(),
        "demo_1_99_percentile_coverage_mean": float(np.mean(coverage)),
        "normalized_wasserstein_per_dimension": wasserstein,
        "normalized_wasserstein_mean": float(np.mean(wasserstein)),
    }


def _plot_band(ax, phase, demo, deployment, title, ylabel) -> None:
    median = np.median(demo, axis=0)
    low, high = np.percentile(demo, [10, 90], axis=0)
    ax.fill_between(phase, low, high, color="#56B4E9", alpha=0.22, linewidth=0)
    ax.plot(phase, median, color="#0072B2", linewidth=1.0, label="Demo median")
    ax.plot(phase, deployment, color="#D55E00", linewidth=1.25, label="Deployment")
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("Normalized task progress (%)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.91", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def _pca_projection(
    demo: np.ndarray, deployment: np.ndarray, components: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardized PCA on demonstrations and project both domains."""
    flat = demo.reshape(-1, demo.shape[-1])
    center = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (flat - center) / scale
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    count = min(components, vt.shape[0])
    basis = vt[:count].T
    demo_projected = ((demo - center) / scale) @ basis
    deployment_projected = ((deployment - center) / scale) @ basis
    variance = singular_values**2 / max(len(standardized) - 1, 1)
    explained = variance[:count] / max(float(np.sum(variance)), 1e-12)
    return demo_projected, deployment_projected, explained, center, scale


def _phase_mahalanobis(
    demo: np.ndarray, deployment: np.ndarray, components: int = 6
) -> tuple[np.ndarray, float]:
    demo_pca, deploy_pca, _, _, _ = _pca_projection(demo, deployment, components)
    dimensions = demo_pca.shape[-1]
    distances = np.zeros(demo_pca.shape[1], dtype=np.float64)
    for phase_index in range(demo_pca.shape[1]):
        samples = demo_pca[:, phase_index]
        mean = samples.mean(axis=0)
        covariance = np.cov(samples, rowvar=False)
        covariance = np.atleast_2d(covariance)
        average_variance = max(float(np.trace(covariance)) / dimensions, 1e-6)
        # Shrinkage makes the local covariance invertible with only 20 demos.
        regularized = 0.85 * covariance + 0.15 * average_variance * np.eye(dimensions)
        delta = deploy_pca[phase_index] - mean
        distances[phase_index] = float(
            np.sqrt(max(delta @ np.linalg.pinv(regularized) @ delta, 0.0))
        )
    # Wilson-Hilferty approximation for sqrt(chi2_0.95).
    z95 = 1.6448536269514722
    chi_square_95 = dimensions * (
        1 - 2 / (9 * dimensions) + z95 * np.sqrt(2 / (9 * dimensions))
    ) ** 3
    return distances, float(np.sqrt(chi_square_95))


def _dtw_distance(
    reference: np.ndarray, query: np.ndarray, window_fraction: float = 0.15
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    n, m = len(reference), len(query)
    window = max(abs(n - m), int(max(n, m) * window_fraction))
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        lower, upper = max(1, i - window), min(m, i + window)
        for j in range(lower, upper + 1):
            cost = float(np.linalg.norm(reference[i - 1] - query[j - 1]))
            current[j] = cost + min(current[j - 1], previous[j], previous[j - 1])
        previous = current
    return float(previous[m] / max(n + m, 1))


def _tactile_features(tactile: np.ndarray) -> np.ndarray:
    tactile = np.asarray(tactile, dtype=np.float64)
    total = tactile.sum(axis=(-2, -1))
    maximum = tactile.max(axis=(-2, -1))
    mean = tactile.mean(axis=(-2, -1))
    threshold = max(0.1 * float(np.max(tactile)), 1e-6)
    area = (tactile >= threshold).mean(axis=(-2, -1))
    row_coordinates = np.arange(tactile.shape[-2])[None, None, :, None]
    col_coordinates = np.arange(tactile.shape[-1])[None, None, None, :]
    denominator = np.maximum(total, 1e-12)
    cop_row = (tactile * row_coordinates).sum(axis=(-2, -1)) / denominator
    cop_col = (tactile * col_coordinates).sum(axis=(-2, -1)) / denominator
    row_profile = tactile.mean(axis=-1)
    col_profile = tactile.mean(axis=-2)
    return np.concatenate(
        [
            mean[..., None], maximum[..., None], area[..., None],
            cop_row[..., None], cop_col[..., None],
            row_profile, col_profile,
        ],
        axis=-1,
    )


def _multimodal_features(
    demo_state: np.ndarray,
    demo_action: np.ndarray,
    deploy_state: np.ndarray,
    deploy_action: np.ndarray,
    demo_imu: np.ndarray | None,
    deploy_imu: np.ndarray | None,
    demo_tactile: np.ndarray | None,
    deploy_tactile: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    demo_parts = [demo_state, demo_action]
    deploy_parts = [deploy_state, deploy_action]
    modalities = ["state", "action"]
    if demo_imu is not None and deploy_imu is not None:
        demo_parts.append(demo_imu)
        deploy_parts.append(deploy_imu)
        modalities.append("imu")
    if demo_tactile is not None and deploy_tactile is not None:
        demo_parts.append(_tactile_features(demo_tactile))
        deploy_parts.append(_tactile_features(deploy_tactile[None, ...])[0])
        modalities.append("tactile")
    return (
        np.concatenate(demo_parts, axis=-1),
        np.concatenate(deploy_parts, axis=-1),
        modalities,
    )


def _plot_3d_manifold(
    ax,
    demo_projected: np.ndarray,
    deploy_projected: np.ndarray,
    explained: np.ndarray,
    nearest_episode: int,
    title: str,
    phase: np.ndarray,
) -> None:
    for episode in range(len(demo_projected)):
        color = "#009E73" if episode == nearest_episode else "#56B4E9"
        alpha = 0.9 if episode == nearest_episode else 0.16
        width = 1.35 if episode == nearest_episode else 0.55
        ax.plot(
            demo_projected[episode, :, 0],
            demo_projected[episode, :, 1],
            demo_projected[episode, :, 2],
            color=color, alpha=alpha, linewidth=width,
        )
    median_path = np.median(demo_projected, axis=0)
    ax.plot(
        median_path[:, 0], median_path[:, 1], median_path[:, 2],
        color="#0072B2", linewidth=2.0, label="Demo median",
    )
    stride = max(len(deploy_projected) // 80, 1)
    points = ax.scatter(
        deploy_projected[::stride, 0],
        deploy_projected[::stride, 1],
        deploy_projected[::stride, 2],
        c=phase[::stride], cmap="plasma", s=12, depthshade=False,
        label="Deployment",
    )
    ax.plot(
        deploy_projected[:, 0], deploy_projected[:, 1], deploy_projected[:, 2],
        color="#D55E00", linewidth=1.1, alpha=0.75,
    )
    ax.scatter(*deploy_projected[0], color="#009E73", marker="o", s=32)
    ax.scatter(*deploy_projected[-1], color="#D55E00", marker="^", s=38)
    variance_text = ", ".join(f"{value * 100:.0f}%" for value in explained[:3])
    ax.set_title(
        f"{title}\nPCA explained variance: {variance_text}",
        loc="left", fontweight="semibold", fontsize=8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(elev=25, azim=-52)
    ax.grid(True, color="0.9", linewidth=0.4)
    return points


def _manifold_comparison(
    output: Path,
    demo_state: np.ndarray,
    demo_action: np.ndarray,
    deploy_state: np.ndarray,
    deploy_action: np.ndarray,
    demo_imu: np.ndarray | None,
    deploy_imu: np.ndarray | None,
    demo_tactile: np.ndarray | None,
    deploy_tactile: np.ndarray | None,
    nearest_episode: int,
    plt,
) -> dict[str, Any]:
    phase = np.linspace(0, 100, len(deploy_state))
    joint_action_demo = np.concatenate([demo_state, demo_action], axis=-1)
    joint_action_deploy = np.concatenate([deploy_state, deploy_action], axis=-1)
    multimodal_demo, multimodal_deploy, modalities = _multimodal_features(
        demo_state, demo_action, deploy_state, deploy_action,
        demo_imu, deploy_imu, demo_tactile, deploy_tactile,
    )
    spaces = (
        ("Joint-state manifold", demo_state, deploy_state),
        ("Action manifold", demo_action, deploy_action),
        ("State-action manifold", joint_action_demo, joint_action_deploy),
        ("Multimodal behavior manifold", multimodal_demo, multimodal_deploy),
    )
    projections = []
    metrics: dict[str, Any] = {"multimodal_features": modalities}
    for name, demo, deploy in spaces:
        demo_pca, deploy_pca, explained, _, _ = _pca_projection(demo, deploy, 3)
        projections.append((name, demo_pca, deploy_pca, explained))
        metrics[name] = {
            "explained_variance": explained.tolist(),
            "deployment_path_length": float(
                np.linalg.norm(np.diff(deploy_pca, axis=0), axis=1).sum()
            ),
            "demo_path_length_median": float(
                np.median(
                    np.linalg.norm(np.diff(demo_pca, axis=1), axis=2).sum(axis=1)
                )
            ),
        }

    mahalanobis, threshold = _phase_mahalanobis(
        multimodal_demo, multimodal_deploy, components=6
    )
    phase_mean = multimodal_demo.mean(axis=0)
    phase_std = multimodal_demo.std(axis=0)
    z_score = (multimodal_deploy - phase_mean) / np.maximum(phase_std, 1e-6)
    feature_deviation = np.sqrt(np.mean(np.square(z_score), axis=1))

    standardized_demo = (
        joint_action_demo - joint_action_demo.reshape(-1, joint_action_demo.shape[-1]).mean(axis=0)
    )
    scale = joint_action_demo.reshape(-1, joint_action_demo.shape[-1]).std(axis=0)
    standardized_demo /= np.maximum(scale, 1e-8)
    standardized_deploy = (
        joint_action_deploy
        - joint_action_demo.reshape(-1, joint_action_demo.shape[-1]).mean(axis=0)
    ) / np.maximum(scale, 1e-8)
    dtw_points = min(120, len(deploy_state))
    deploy_dtw = _resample(standardized_deploy, dtw_points)
    dtw_distances = np.asarray(
        [
            _dtw_distance(_resample(episode, dtw_points), deploy_dtw)
            for episode in standardized_demo
        ]
    )
    dtw_nearest = int(np.argmin(dtw_distances))

    fig = plt.figure(figsize=(13.2, 11.4))
    fig.subplots_adjust(
        left=0.055, right=0.965, bottom=0.055, top=0.965,
        hspace=0.62, wspace=0.62,
    )
    grid = fig.add_gridspec(3, 12, height_ratios=[1.25, 1.25, 0.95])
    positions = ((0, slice(0, 6)), (0, slice(6, 12)), (1, slice(0, 6)), (1, slice(6, 12)))
    for (name, demo_pca, deploy_pca, explained), (row, columns) in zip(
        projections, positions
    ):
        ax = fig.add_subplot(grid[row, columns], projection="3d")
        points = _plot_3d_manifold(
            ax, demo_pca, deploy_pca, explained, nearest_episode, name, phase
        )
        fig.colorbar(points, ax=ax, shrink=0.55, pad=0.08, label="Task progress (%)")

    ax_heat = fig.add_subplot(grid[2, :4])
    state_action_z = (
        joint_action_deploy - joint_action_demo.mean(axis=0)
    ) / np.maximum(joint_action_demo.std(axis=0), 1e-6)
    heat = ax_heat.imshow(
        state_action_z.T, cmap="coolwarm", vmin=-3, vmax=3,
        aspect="auto", origin="lower", extent=[0, 100, 0, 13],
        interpolation="nearest",
    )
    ax_heat.set_yticks(
        np.arange(13) + 0.5,
        [f"q{i + 1}" for i in range(6)] + [f"a{i + 1}" for i in range(6)] + ["grip"],
        fontsize=6,
    )
    ax_heat.set_title("Phase-conditioned deviation map", loc="left", fontweight="semibold")
    ax_heat.set_xlabel("Task progress (%)")
    ax_heat.set_ylabel("State/action dimension")
    fig.colorbar(heat, ax=ax_heat, fraction=0.04, pad=0.025, label="z-score")

    ax_mahal = fig.add_subplot(grid[2, 5:9])
    ax_mahal.plot(phase, mahalanobis, color="#CC79A7", linewidth=1.25)
    ax_mahal.fill_between(
        phase, 0, mahalanobis, where=mahalanobis > threshold,
        color="#D55E00", alpha=0.18, label="Outside 95% region",
    )
    ax_mahal.axhline(
        threshold, color="#D55E00", linestyle="--", label="Approx. 95% threshold"
    )
    ax_mahal.set_title(
        "Phase-local multimodal distance", loc="left", fontweight="semibold"
    )
    ax_mahal.set_xlabel("Task progress (%)")
    ax_mahal.set_ylabel("Regularized Mahalanobis distance")
    ax_mahal.legend(frameon=False, fontsize=7)

    ax_dtw = fig.add_subplot(grid[2, 10:])
    colors = np.full(len(dtw_distances), "#56B4E9", dtype=object)
    colors[dtw_nearest] = "#009E73"
    ax_dtw.bar(np.arange(len(dtw_distances)), dtw_distances, color=colors)
    ax_dtw.set_title("DTW distance to each demo", loc="left", fontweight="semibold")
    ax_dtw.set_xlabel("Demonstration episode")
    ax_dtw.set_ylabel("Windowed DTW distance")
    ax_dtw.set_xticks(np.arange(len(dtw_distances)))
    ax_dtw.tick_params(axis="x", labelsize=6)
    for ax in (ax_mahal, ax_dtw):
        ax.grid(True, color="0.91", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison_manifold_3d.png", dpi=300, facecolor="white")
    plt.close(fig)

    metrics.update(
        {
            "mahalanobis_95_threshold": threshold,
            "mahalanobis_mean": float(np.mean(mahalanobis)),
            "mahalanobis_max": float(np.max(mahalanobis)),
            "mahalanobis_outside_fraction": float(np.mean(mahalanobis > threshold)),
            "phase_deviation_rms_mean": float(np.mean(feature_deviation)),
            "phase_deviation_rms_max": float(np.max(feature_deviation)),
            "dtw_distances": dtw_distances.tolist(),
            "dtw_nearest_episode": dtw_nearest,
            "dtw_nearest_distance": float(dtw_distances[dtw_nearest]),
        }
    )
    return metrics


def _rank_bins(values: np.ndarray, bins: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    return np.minimum((ranks / max(len(values), 1) * bins).astype(int), bins - 1)


def _normalized_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    x_bin, y_bin = _rank_bins(x, bins), _rank_bins(y, bins)
    joint = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(joint, (x_bin, y_bin), 1)
    joint /= max(float(joint.sum()), 1.0)
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    expected = px[:, None] * py[None, :]
    valid = (joint > 0) & (expected > 0)
    mutual_information = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return mutual_information / max(np.sqrt(hx * hy), 1e-12)


def _nmi_matrix(signals: np.ndarray) -> np.ndarray:
    dimensions = signals.shape[1]
    result = np.eye(dimensions, dtype=np.float64)
    for first in range(dimensions):
        for second in range(first + 1, dimensions):
            value = _normalized_mutual_information(
                signals[:, first], signals[:, second]
            )
            result[first, second] = result[second, first] = value
    return result


def _regularized_cca(
    first: np.ndarray, second: np.ndarray, components: int = 3, ridge: float = 1e-3
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = (first - first.mean(axis=0)) / np.maximum(first.std(axis=0), 1e-8)
    second = (second - second.mean(axis=0)) / np.maximum(second.std(axis=0), 1e-8)
    count = max(len(first) - 1, 1)
    covariance_x = first.T @ first / count + ridge * np.eye(first.shape[1])
    covariance_y = second.T @ second / count + ridge * np.eye(second.shape[1])
    cross = first.T @ second / count

    def inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        return (vectors * (1.0 / np.sqrt(np.maximum(values, ridge)))) @ vectors.T

    whitened = inverse_sqrt(covariance_x) @ cross @ inverse_sqrt(covariance_y)
    correlations = np.linalg.svd(whitened, compute_uv=False)
    return np.clip(correlations[:components], 0.0, 1.0)


def _cca_matrix(blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, list[float]]]:
    names = list(blocks)
    matrix = np.eye(len(names), dtype=np.float64)
    spectra: dict[str, list[float]] = {}
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            correlations = _regularized_cca(blocks[names[first]], blocks[names[second]])
            matrix[first, second] = matrix[second, first] = float(correlations[0])
            spectra[f"{names[first]}__{names[second]}"] = correlations.tolist()
    return matrix, spectra


def _standardize_signal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    scale = np.percentile(values, 75) - np.percentile(values, 25)
    return (values - median) / max(float(scale), 1e-8)


def _summary_signals(
    state: np.ndarray,
    action: np.ndarray,
    imu: np.ndarray | None,
    tactile: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    # The phase derivative is used consistently for demos and deployment.
    state_motion = np.linalg.norm(np.gradient(state, axis=-2), axis=-1)
    action_motion = np.linalg.norm(np.gradient(action[..., :6], axis=-2), axis=-1)
    gripper_motion = np.abs(np.gradient(action[..., 6], axis=-1))
    signals = [state_motion, action_motion, gripper_motion]
    names = ["State motion", "Action motion", "Gripper motion"]
    if imu is not None:
        signals.extend(
            [
                np.linalg.norm(imu[..., 4:7], axis=-1),
                np.abs(np.linalg.norm(imu[..., 7:10], axis=-1) - 1.0),
            ]
        )
        names.extend(["IMU angular", "IMU acceleration"])
    if tactile is not None:
        signals.extend(
            [
                tactile.max(axis=(-2, -1)),
                (tactile >= max(0.1 * float(np.max(tactile)), 1e-6)).mean(
                    axis=(-2, -1)
                ),
            ]
        )
        names.extend(["Tactile peak", "Tactile area"])
    return np.stack(signals, axis=-1), names


def _peak_lag_matrix(signals: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(signals, dtype=np.float64)
    dimensions = signals.shape[1]
    lag_matrix = np.zeros((dimensions, dimensions), dtype=np.float64)
    correlation_matrix = np.eye(dimensions, dtype=np.float64)
    standardized = np.column_stack(
        [_standardize_signal(signals[:, index]) for index in range(dimensions)]
    )
    for first in range(dimensions):
        for second in range(first + 1, dimensions):
            best_correlation, best_lag = 0.0, 0
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    x, y = standardized[-lag:, first], standardized[:lag, second]
                elif lag > 0:
                    x, y = standardized[:-lag, first], standardized[lag:, second]
                else:
                    x, y = standardized[:, first], standardized[:, second]
                if len(x) < 5 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    continue
                correlation = float(np.corrcoef(x, y)[0, 1])
                if abs(correlation) > abs(best_correlation):
                    best_correlation, best_lag = correlation, lag
            lag_matrix[first, second] = best_lag
            lag_matrix[second, first] = -best_lag
            correlation_matrix[first, second] = correlation_matrix[second, first] = best_correlation
    return lag_matrix, correlation_matrix


def _rolling_lag_surface(
    source: np.ndarray,
    target: np.ndarray,
    *,
    window: int = 35,
    max_lag: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = _standardize_signal(source)
    target = _standardize_signal(target)
    centers = np.arange(window // 2, len(source) - window // 2, max(window // 5, 1))
    lags = np.arange(-max_lag, max_lag + 1)
    surface = np.full((len(centers), len(lags)), np.nan)
    half = window // 2
    for row, center in enumerate(centers):
        start, end = center - half, center + half
        for column, lag in enumerate(lags):
            shifted_start, shifted_end = start + lag, end + lag
            if shifted_start < 0 or shifted_end > len(target):
                continue
            x, y = source[start:end], target[shifted_start:shifted_end]
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                surface[row, column] = np.corrcoef(x, y)[0, 1]
    return centers, lags, surface


def _plot_coupling_network(
    ax, matrix: np.ndarray, difference: np.ndarray, names: list[str]
) -> None:
    count = len(names)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + np.pi / 2
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    for first in range(count):
        for second in range(first + 1, count):
            strength = matrix[first, second]
            if strength < 0.08:
                continue
            delta = difference[first, second]
            color = "#D55E00" if delta > 0 else "#0072B2"
            ax.plot(
                positions[[first, second], 0], positions[[first, second], 1],
                color=color, linewidth=0.5 + 7.0 * strength,
                alpha=min(0.25 + strength, 0.9),
            )
    ax.scatter(
        positions[:, 0], positions[:, 1], s=420, color="#F4F6F7",
        edgecolor="0.25", linewidth=0.8, zorder=3,
    )
    for position, name in zip(positions, names):
        ax.text(*position, name.replace(" ", "\n"), ha="center", va="center", fontsize=6)
    ax.text(
        -1.2, -1.22, "Orange: stronger in deployment   Blue: weaker in deployment",
        fontsize=6.5,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.3, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Cross-modal coupling network", loc="left", fontweight="semibold")


def _cross_modal_comparison(
    output: Path,
    demo_state: np.ndarray,
    demo_action: np.ndarray,
    deploy_state: np.ndarray,
    deploy_action: np.ndarray,
    demo_imu: np.ndarray | None,
    deploy_imu: np.ndarray | None,
    demo_tactile: np.ndarray | None,
    deploy_tactile: np.ndarray | None,
    plt,
) -> dict[str, Any]:
    demo_signals, signal_names = _summary_signals(
        demo_state, demo_action, demo_imu, demo_tactile
    )
    deploy_signals, _ = _summary_signals(
        deploy_state, deploy_action, deploy_imu, deploy_tactile
    )
    demo_nmi_episodes = np.stack([_nmi_matrix(episode) for episode in demo_signals])
    demo_nmi = np.median(demo_nmi_episodes, axis=0)
    deploy_nmi = _nmi_matrix(deploy_signals)
    nmi_difference = deploy_nmi - demo_nmi

    demo_blocks = {
        "State": demo_state.reshape(-1, demo_state.shape[-1]),
        "Action": demo_action.reshape(-1, demo_action.shape[-1]),
    }
    deploy_blocks = {"State": deploy_state, "Action": deploy_action}
    if demo_imu is not None and deploy_imu is not None:
        demo_blocks["IMU"] = demo_imu[..., 4:10].reshape(-1, 6)
        deploy_blocks["IMU"] = deploy_imu[..., 4:10]
    if demo_tactile is not None and deploy_tactile is not None:
        demo_touch_features = _tactile_features(demo_tactile)
        deploy_touch_features = _tactile_features(deploy_tactile[None, ...])[0]
        demo_blocks["Tactile"] = demo_touch_features.reshape(
            -1, demo_touch_features.shape[-1]
        )
        deploy_blocks["Tactile"] = deploy_touch_features
    demo_cca, demo_spectra = _cca_matrix(demo_blocks)
    deploy_cca, deploy_spectra = _cca_matrix(deploy_blocks)
    cca_difference = deploy_cca - demo_cca
    block_names = list(demo_blocks)

    demo_lags = []
    demo_lag_correlations = []
    max_lag = max(len(deploy_signals) // 12, 2)
    for episode in demo_signals:
        lag, correlation = _peak_lag_matrix(episode, max_lag)
        demo_lags.append(lag)
        demo_lag_correlations.append(correlation)
    deploy_lag, deploy_lag_correlation = _peak_lag_matrix(deploy_signals, max_lag)
    demo_lag = np.median(np.stack(demo_lags), axis=0)
    demo_lag_correlation = np.median(np.stack(demo_lag_correlations), axis=0)

    fig = plt.figure(figsize=(13.2, 13.0))
    fig.subplots_adjust(
        left=0.055, right=0.965, bottom=0.055, top=0.965,
        hspace=0.64, wspace=0.72,
    )
    grid = fig.add_gridspec(4, 12, height_ratios=[1.0, 1.0, 1.15, 1.0])
    matrices = (
        ("Demo normalized mutual information", demo_nmi, "viridis", 0, 1),
        ("Deployment normalized mutual information", deploy_nmi, "viridis", 0, 1),
        ("Coupling difference (deploy − demo)", nmi_difference, "coolwarm", -0.5, 0.5),
    )
    for (title, matrix, cmap, vmin, vmax), columns in zip(
        matrices, (slice(0, 4), slice(4, 8), slice(8, 12))
    ):
        ax = fig.add_subplot(grid[0, columns])
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(signal_names)), signal_names, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(signal_names)), signal_names, fontsize=6)
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)

    cca_matrices = (
        ("Demo regularized CCA", demo_cca, 0, 1),
        ("Deployment regularized CCA", deploy_cca, 0, 1),
        ("CCA difference (deploy − demo)", cca_difference, -0.5, 0.5),
    )
    for (title, matrix, vmin, vmax), columns in zip(
        cca_matrices, (slice(0, 4), slice(4, 8), slice(8, 12))
    ):
        ax = fig.add_subplot(grid[1, columns])
        image = ax.imshow(
            matrix, cmap="coolwarm" if vmin < 0 else "magma", vmin=vmin, vmax=vmax
        )
        ax.set_xticks(np.arange(len(block_names)), block_names, fontsize=7)
        ax.set_yticks(np.arange(len(block_names)), block_names, fontsize=7)
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)

    ax_signals = fig.add_subplot(grid[2, :5])
    phase = np.linspace(0, 100, len(deploy_signals))
    for index, name in enumerate(signal_names):
        signal = _standardize_signal(deploy_signals[:, index])
        signal = np.clip(signal, -3, 5) + index * 5.0
        ax_signals.plot(phase, signal, linewidth=0.9, label=name)
    ax_signals.set_yticks(
        np.arange(len(signal_names)) * 5.0, signal_names, fontsize=7
    )
    ax_signals.set_title(
        "Deployment cross-modal event raster", loc="left", fontweight="semibold"
    )
    ax_signals.set_xlabel("Task progress (%)")
    ax_signals.set_ylabel("Robust standardized signals (offset)")

    ax_network = fig.add_subplot(grid[2, 5:9])
    _plot_coupling_network(ax_network, deploy_nmi, nmi_difference, signal_names)

    ax_lag = fig.add_subplot(grid[2, 9:])
    lag_difference = deploy_lag - demo_lag
    lag_image = ax_lag.imshow(
        lag_difference / len(deploy_signals) * 100.0,
        cmap="coolwarm", vmin=-15, vmax=15,
    )
    ax_lag.set_xticks(np.arange(len(signal_names)), signal_names, rotation=45, ha="right", fontsize=6)
    ax_lag.set_yticks(np.arange(len(signal_names)), signal_names, fontsize=6)
    ax_lag.set_title(
        "Optimal-lag shift (deploy − demo)", loc="left", fontweight="semibold",
        fontsize=8,
    )
    fig.colorbar(lag_image, ax=ax_lag, fraction=0.05, pad=0.03, label="Task progress (%)")

    source = deploy_signals[:, 1]  # action motion
    surface_targets = []
    if "IMU angular" in signal_names:
        surface_targets.append(("Action ↔ IMU angular", deploy_signals[:, signal_names.index("IMU angular")]))
    if "Tactile peak" in signal_names:
        surface_targets.append(("Action ↔ tactile", deploy_signals[:, signal_names.index("Tactile peak")]))
    if not surface_targets:
        surface_targets.append(("State ↔ action", deploy_signals[:, 0]))
    while len(surface_targets) < 2:
        surface_targets.append(("Action ↔ gripper", deploy_signals[:, 2]))
    for (title, target), columns in zip(surface_targets[:2], (slice(0, 6), slice(6, 12))):
        ax = fig.add_subplot(grid[3, columns], projection="3d")
        centers, lags, surface = _rolling_lag_surface(source, target)
        center_phase = centers / max(len(source) - 1, 1) * 100.0
        lag_phase = lags / max(len(source) - 1, 1) * 100.0
        x_mesh, y_mesh = np.meshgrid(center_phase, lag_phase, indexing="ij")
        plotted = ax.plot_surface(
            x_mesh, y_mesh, np.nan_to_num(surface),
            cmap="coolwarm", vmin=-1, vmax=1, linewidth=0, antialiased=True,
        )
        ax.set_title(
            f"Dynamic lagged coupling surface: {title}",
            loc="left", fontweight="semibold", fontsize=8,
        )
        ax.set_xlabel("Task progress (%)")
        ax.set_ylabel("Target lag (%)")
        ax.set_zlabel("Correlation")
        ax.set_zlim(-1, 1)
        ax.view_init(elev=28, azim=-58)
        fig.colorbar(plotted, ax=ax, shrink=0.58, pad=0.08)

    for ax in fig.axes:
        if getattr(ax, "name", "") != "3d" and not ax.get_images():
            ax.grid(True, color="0.91", linewidth=0.55)
            if hasattr(ax, "spines"):
                ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison_cross_modal.png", dpi=300, facecolor="white")
    plt.close(fig)

    strongest_pair = None
    strongest_delta = -np.inf
    for first in range(len(signal_names)):
        for second in range(first + 1, len(signal_names)):
            if abs(nmi_difference[first, second]) > strongest_delta:
                strongest_delta = abs(nmi_difference[first, second])
                strongest_pair = [signal_names[first], signal_names[second]]
    return {
        "signal_names": signal_names,
        "block_names": block_names,
        "demo_nmi": demo_nmi.tolist(),
        "deployment_nmi": deploy_nmi.tolist(),
        "nmi_difference": nmi_difference.tolist(),
        "demo_cca": demo_cca.tolist(),
        "deployment_cca": deploy_cca.tolist(),
        "cca_difference": cca_difference.tolist(),
        "demo_cca_spectra": demo_spectra,
        "deployment_cca_spectra": deploy_spectra,
        "demo_optimal_lag_steps": demo_lag.tolist(),
        "deployment_optimal_lag_steps": deploy_lag.tolist(),
        "demo_peak_lag_correlation": demo_lag_correlation.tolist(),
        "deployment_peak_lag_correlation": deploy_lag_correlation.tolist(),
        "strongest_nmi_change_pair": strongest_pair,
        "strongest_nmi_change_abs": float(strongest_delta),
        "mean_abs_nmi_change": float(np.mean(np.abs(nmi_difference))),
        "mean_abs_cca_change": float(np.mean(np.abs(cca_difference))),
    }


def _decode_image(cell) -> np.ndarray | None:
    import cv2

    value = cell.as_py()
    encoded = value.get("bytes") if isinstance(value, dict) else None
    if not encoded:
        return None
    bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _visual_comparison(
    run_dir: Path,
    demo_path: Path,
    output: Path,
    deployment_length: int,
) -> bool:
    import matplotlib.pyplot as plt

    if not (run_dir / "videos").is_dir():
        return False
    table = pq.read_table(
        demo_path, columns=["observation.image", "observation.wrist_image"]
    )
    phases = np.linspace(0.0, 1.0, 5)
    demo_indices = np.rint(phases * (len(table) - 1)).astype(int)
    deploy_indices = np.rint(phases * (deployment_length - 1)).astype(int)
    lengths = _chunk_lengths(run_dir)
    fig, axes = plt.subplots(
        4, 5, figsize=(12, 7.1),
        gridspec_kw={"wspace": 0.025, "hspace": 0.055},
    )
    for column, (phase, demo_index, deploy_index) in enumerate(
        zip(phases, demo_indices, deploy_indices)
    ):
        images = (
            _decode_image(table["observation.image"][demo_index]),
            _decode_image(table["observation.wrist_image"][demo_index]),
            _read_segment_frame(run_dir, "cam_high", int(deploy_index), lengths),
            _read_segment_frame(run_dir, "cam_wrist", int(deploy_index), lengths),
        )
        for row, image in enumerate(images):
            ax = axes[row, column]
            if image is not None:
                ax.imshow(image)
            else:
                ax.set_facecolor("0.92")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.25")
        axes[0, column].set_title(f"{phase * 100:.0f}%", fontsize=8)
    for row, label in enumerate(
        ("Demo scene", "Demo wrist", "Deploy scene", "Deploy wrist")
    ):
        axes[row, 0].set_ylabel(label, fontsize=8)
    fig.savefig(output / "comparison_images.png", dpi=300, facecolor="white")
    plt.close(fig)
    return True


def compare(
    run_dir: Path,
    dataset_root: Path,
    output_dir: Path | None = None,
    *,
    phase_points: int = 200,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    dataset_root = Path(dataset_root)
    metadata, deployment_raw = load_run(run_dir)
    episodes = load_demonstrations(dataset_root)
    output = (
        Path(output_dir)
        if output_dir is not None
        else run_dir / "comparison_with_demonstrations"
    )
    output.mkdir(parents=True, exist_ok=True)
    _style()

    demo_state = _phase_stack(episodes, "observation.state", phase_points)
    demo_action = _phase_stack(episodes, "action", phase_points)
    if demo_state is None or demo_action is None:
        raise ValueError("示教数据缺少observation.state或action")
    deploy_state = _resample(deployment_raw["joint_position"], phase_points)
    deploy_action = _resample(deployment_raw["safe_action"], phase_points)
    state_trajectory, nearest_state = _trajectory_metrics(demo_state, deploy_state)
    action_trajectory, nearest_action = _trajectory_metrics(demo_action, deploy_action)
    nearest_episode = nearest_state

    metrics: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "demonstration_episodes": len(episodes),
        "phase_points": phase_points,
        "state": {
            **state_trajectory,
            **_distribution_metrics(demo_state, deploy_state),
        },
        "action": {
            **action_trajectory,
            **_distribution_metrics(demo_action, deploy_action),
        },
        "nearest_episode_by_state": nearest_state,
        "nearest_episode_by_action": nearest_action,
    }

    phase = np.linspace(0, 100, phase_points)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
    for joint, ax in enumerate(axes.flat):
        _plot_band(
            ax, phase, demo_state[:, :, joint], deploy_state[:, joint],
            f"Joint {joint + 1}", "Position (rad)",
        )
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.subplots_adjust(hspace=0.42, wspace=0.3)
    fig.savefig(output / "comparison_joints.png", dpi=300, facecolor="white")
    plt.close(fig)

    demo_time = [
        np.asarray(episode.get("timestamp", np.arange(len(episode["action"])) / 50.0))
        for episode in episodes
    ]
    deploy_time = np.asarray(deployment_raw["time_s"])
    demo_velocity_rms = [
        _derivative_rms(episode["observation.state"], time, 1)
        for episode, time in zip(episodes, demo_time)
    ]
    demo_accel_rms = [
        _derivative_rms(episode["observation.state"], time, 2)
        for episode, time in zip(episodes, demo_time)
    ]
    demo_jerk_rms = [
        _derivative_rms(episode["observation.state"], time, 3)
        for episode, time in zip(episodes, demo_time)
    ]
    deploy_smoothness = {
        "velocity_rms": _derivative_rms(
            deployment_raw["joint_position"], deploy_time, 1
        ),
        "acceleration_rms": _derivative_rms(
            deployment_raw["joint_position"], deploy_time, 2
        ),
        "jerk_rms": _derivative_rms(
            deployment_raw["joint_position"], deploy_time, 3
        ),
    }
    metrics["smoothness"] = {}
    for name, reference in (
        ("velocity_rms", demo_velocity_rms),
        ("acceleration_rms", demo_accel_rms),
        ("jerk_rms", demo_jerk_rms),
    ):
        metrics["smoothness"][name] = {
            "deployment": deploy_smoothness[name],
            "demo_median": float(np.median(reference)),
            "demo_p10": float(np.percentile(reference, 10)),
            "demo_p90": float(np.percentile(reference, 90)),
            "deployment_percentile": _percentile_rank(reference, deploy_smoothness[name]),
        }

    fig = plt.figure(figsize=(12, 7.4))
    grid = fig.add_gridspec(2, 12, hspace=0.5, wspace=0.65)
    ax_gripper = fig.add_subplot(grid[0, :6])
    _plot_band(
        ax_gripper, phase, demo_action[:, :, 6], deploy_action[:, 6],
        "Gripper action", "Position (rad)",
    )
    ax_gripper.legend(frameon=False, ncol=2)

    ax_coverage = fig.add_subplot(grid[0, 6:])
    labels = [f"J{i + 1}" for i in range(6)] + ["Grip"]
    x = np.arange(7)
    action_coverage = np.asarray(
        metrics["action"]["demo_1_99_percentile_coverage_per_dimension"]
    )
    ax_coverage.bar(x, action_coverage * 100, color="#0072B2", alpha=0.85)
    ax_coverage.axhline(95, color="#D55E00", linestyle="--", label="95%")
    ax_coverage.set_xticks(x, labels)
    ax_coverage.set_ylim(0, 105)
    ax_coverage.set_ylabel("Deployment inside demo range (%)")
    ax_coverage.set_title("Demonstration-distribution coverage", loc="left", fontweight="semibold")
    ax_coverage.legend(frameon=False)

    ax_smooth = fig.add_subplot(grid[1, :5])
    smooth_labels = ("Velocity", "Acceleration", "Jerk")
    demo_medians = [
        metrics["smoothness"][key]["demo_median"]
        for key in ("velocity_rms", "acceleration_rms", "jerk_rms")
    ]
    deployment_values = [
        metrics["smoothness"][key]["deployment"]
        for key in ("velocity_rms", "acceleration_rms", "jerk_rms")
    ]
    ratios = np.asarray(deployment_values) / np.maximum(demo_medians, 1e-12)
    ax_smooth.bar(smooth_labels, ratios, color=("#56B4E9", "#009E73", "#E69F00"))
    ax_smooth.axhline(1.0, color="0.25", linestyle="--", label="Demo median")
    ax_smooth.set_ylabel("Deployment / demo median")
    ax_smooth.set_title("Motion smoothness", loc="left", fontweight="semibold")
    ax_smooth.legend(frameon=False)

    ax_distance = fig.add_subplot(grid[1, 5:9])
    distances = np.asarray(
        metrics["action"]["normalized_wasserstein_per_dimension"]
    )
    ax_distance.bar(labels, distances, color="#CC79A7")
    ax_distance.set_ylabel("Normalized distance")
    ax_distance.set_title("Action distribution shift", loc="left", fontweight="semibold")

    ax_summary = fig.add_subplot(grid[1, 9:])
    ax_summary.axis("off")
    summary_lines = [
        f"Demo episodes        {len(episodes)}",
        f"Nearest episode      {nearest_episode:02d}",
        f"State phase coverage {metrics['state']['phase_band_coverage_mean'] * 100:.1f}%",
        f"Action range coverage {metrics['action']['demo_1_99_percentile_coverage_mean'] * 100:.1f}%",
        f"Action dist. shift   {metrics['action']['normalized_wasserstein_mean']:.2f}",
        f"Velocity percentile  {metrics['smoothness']['velocity_rms']['deployment_percentile']:.0f}%",
        f"Jerk percentile      {metrics['smoothness']['jerk_rms']['deployment_percentile']:.0f}%",
    ]
    ax_summary.text(
        0.03, 0.97, "\n".join(summary_lines), va="top", family="monospace",
        fontsize=9, linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F4F6F7", "edgecolor": "0.75"},
    )
    for ax in (ax_coverage, ax_smooth, ax_distance):
        ax.grid(True, axis="y", color="0.91", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "comparison_overview.png", dpi=300, facecolor="white")
    plt.close(fig)

    demo_imu = _phase_stack(episodes, "sensor.imu", phase_points)
    demo_tactile = _phase_stack(episodes, "sensor.tactile", phase_points)
    has_imu = demo_imu is not None and "imu" in deployment_raw
    has_tactile = demo_tactile is not None and "tactile" in deployment_raw
    deploy_imu = (
        _resample(deployment_raw["imu"], phase_points) if has_imu else None
    )
    deploy_tactile = (
        _resample(deployment_raw["tactile"], phase_points) if has_tactile else None
    )
    if has_imu or has_tactile:
        rows = int(has_imu) + int(has_tactile)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), squeeze=False)
        row = 0
        if has_imu:
            demo_gyro = np.linalg.norm(demo_imu[:, :, 4:7], axis=2)
            deploy_gyro = np.linalg.norm(deploy_imu[:, 4:7], axis=1)
            demo_accel = np.linalg.norm(demo_imu[:, :, 7:10], axis=2)
            deploy_accel = np.linalg.norm(deploy_imu[:, 7:10], axis=1)
            _plot_band(
                axes[row, 0], phase, demo_gyro, deploy_gyro,
                "Angular-rate magnitude", "rad/s",
            )
            _plot_band(
                axes[row, 1], phase, demo_accel, deploy_accel,
                "Acceleration magnitude", "g",
            )
            metrics["imu"] = {
                "gyro_norm_distribution_distance": _normalized_wasserstein(
                    demo_gyro.ravel(), deploy_gyro
                ),
                "accel_norm_distribution_distance": _normalized_wasserstein(
                    demo_accel.ravel(), deploy_accel
                ),
            }
            row += 1
        if has_tactile:
            demo_touch_max = demo_tactile.max(axis=(2, 3))
            deploy_touch_max = deploy_tactile.max(axis=(1, 2))
            demo_touch_mean = demo_tactile.mean(axis=(2, 3))
            deploy_touch_mean = deploy_tactile.mean(axis=(1, 2))
            _plot_band(
                axes[row, 0], phase, demo_touch_max, deploy_touch_max,
                "Maximum tactile response", "Response",
            )
            _plot_band(
                axes[row, 1], phase, demo_touch_mean, deploy_touch_mean,
                "Mean tactile response", "Response",
            )
            metrics["tactile"] = {
                "maximum_response_distribution_distance": _normalized_wasserstein(
                    demo_touch_max.ravel(), deploy_touch_max
                ),
                "mean_response_distribution_distance": _normalized_wasserstein(
                    demo_touch_mean.ravel(), deploy_touch_mean
                ),
                "demo_peak_median": float(
                    np.median(demo_tactile.max(axis=(1, 2, 3)))
                ),
                "deployment_peak": float(np.max(deploy_tactile)),
            }
        axes[0, 0].legend(frameon=False, ncol=2)
        fig.subplots_adjust(hspace=0.45, wspace=0.28)
        fig.savefig(output / "comparison_multimodal.png", dpi=300, facecolor="white")
        plt.close(fig)

    metrics["manifold_analysis"] = _manifold_comparison(
        output,
        demo_state,
        demo_action,
        deploy_state,
        deploy_action,
        demo_imu if has_imu else None,
        deploy_imu,
        demo_tactile if has_tactile else None,
        deploy_tactile,
        nearest_episode,
        plt,
    )
    metrics["cross_modal_analysis"] = _cross_modal_comparison(
        output,
        demo_state,
        demo_action,
        deploy_state,
        deploy_action,
        demo_imu if has_imu else None,
        deploy_imu,
        demo_tactile if has_tactile else None,
        deploy_tactile,
        plt,
    )
    demo_path = Path(str(episodes[nearest_episode]["_path"].item()))
    metrics["visual_comparison_generated"] = _visual_comparison(
        run_dir, demo_path, output, len(deployment_raw["time_s"])
    )
    metrics["nearest_episode_path"] = str(demo_path.resolve())
    (output / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比ACT真机部署记录与LeRobot示教数据分布"
    )
    parser.add_argument("--run", type=Path, required=True, help="一次策略部署记录目录")
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot示教数据集")
    parser.add_argument("--output", type=Path, help="默认写入run/comparison_with_demonstrations")
    parser.add_argument("--phase-points", type=int, default=200)
    args = parser.parse_args()
    if args.phase_points < 20:
        raise ValueError("--phase-points不能小于20")
    output = compare(
        args.run, args.dataset_root, args.output, phase_points=args.phase_points
    )
    print(f"示教/部署对比结果已保存: {output}")


if __name__ == "__main__":
    main()
