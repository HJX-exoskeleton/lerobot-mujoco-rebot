# reBot ACT MuJoCo 仿真工程

## 1. 目标

`rebot_act_sim` 是独立于旧 `scripts/01_collect_data.py`～`04_deploy_act.py`
的规范化 ACT 仿真流水线。它与 `rebot_act_real` 采用相同的工程原则：

- 采集、训练和部署共享一份数据契约；
- 明确区分机器人反馈状态与下发动作目标；
- 标准 ACT 始终使用双相机和关节反馈；
- IMU、触觉作为辅助字段永久保存在数据集中；
- 只有显式启用多模态训练时，辅助传感器才进入网络；
- checkpoint 保存自身的多模态规格，部署时自动识别；
- 仿真代码不依赖任何真机驱动。

当前任务使用专用场景
`rebot_act_sim/assets/example_scene_rebot_red_cube.xml`。机械臂已由旧版
`asset_rebot/reBot-DevArm_gripper_tactile.xml` 升级为
`asset_rebotarm_b601_colored` 中的官方彩色 B601 模型；任务桌面、纯红色方块、
目标盘、固定场景相机、物体随机范围和成功判定保持不变。

场景通过任务适配文件 `rebot_act_sim/assets/rebotarm_b601_colored_act.xml` 引用
官方目录中的彩色 STL 与 `8×16` 触觉资源，不复制网格。该适配层只做以下兼容：

- 将 B601 基座保持在原任务坐标 `[-0.05, 0, 0.8]`；
- 保留官方 `joint1..joint6`、`finger_left/finger_right` 和位置伺服；
- 使用官方 `end_link` 作为 IK/成功判定末端，使用 `cam_wrist` 作为腕部相机；
- IMU 改读官方 `orientation_wrist/ang_vel_wrist/accel_wrist`；
- 将官方机械臂默认碰撞参数限制在机械臂子树内，避免改变桌面和任务物体碰撞；
- 不引入官方演示场景中的桌面和物体；背景使用官方 `desert.png` 天空盒，原任务
  地面、桌面和灯光参数保持不变。

原杯子网格保持替换为纯红色方块：

```xml
<geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/>
```

MuJoCo 的 box `size` 表示三个方向的半尺寸，因此方块实际边长为 `0.04 m`。
方块质量为 `0.05 kg`，具有自由关节和独立碰撞几何，可以被正常抓取和放置。

由于机械臂外观、夹爪几何、惯量和执行器增益已经升级，旧模型采集的数据集与
checkpoint 不应继续用于正式评估；请使用当前 XML 重新采集、训练，并使用新的
数据集目录或先显式备份旧数据。

## 2. 包结构

```text
rebot_act_sim/
├── configs/act_sim.yaml          # 数据、环境、ACT和训练参数
├── assets/
│   ├── example_scene_rebot_red_cube.xml # 原任务场景
│   └── rebotarm_b601_colored_act.xml    # 官方B601任务适配层
├── config.py                     # 配置与ACT特征构造
├── schema.py                     # 唯一数据契约
├── dataset_writer.py             # 安全创建、追加、保存和丢弃episode
├── environment.py                # MuJoCo任务适配器
├── sensors.py                    # IMU和触觉读取
├── multimodal_policy.py          # IMU MLP、触觉CNN和传感器token融合
├── policy.py                     # 训练/部署统一的策略构造
├── visualization.py              # 同步/异步IMU曲线与触觉热图渲染
├── timing.py                     # WallClockRate 实时墙钟节拍器
├── workflow/
│   ├── collect.py
│   ├── replay.py
│   ├── inspect_dataset.py
│   ├── train.py
│   └── deploy.py
└── tests/
```

## 3. 数据契约

策略基础字段：

```text
observation.image        uint8   [256,256,3]  场景相机
observation.wrist_image  uint8   [256,256,3]  腕部相机
observation.state        float32 [6]          当前六关节反馈，rad
action                   float32 [7]          六关节绝对目标rad + 0/1夹爪
```

辅助字段：

```text
sensor.joint_velocity              float32 [6]
sensor.gripper_feedback            float32 [2]
sensor.imu                         float32 [10]
sensor.tactile_left                float32 [8,16]  策略使用的处理后左触觉
sensor.tactile_right               float32 [8,16]  策略使用的处理后右触觉
sensor.tactile_left_raw            float32 [8,16]  当前帧原始左接近强度
sensor.tactile_right_raw           float32 [8,16]  当前帧原始右接近强度
sensor.sim_time                    float64 [1]
episode.object_initial_position    float32 [6]
```

`sensor.imu` 的顺序是末端 `quaternion(wxyz) + gyro(xyz) + accel(xyz)`。

MuJoCo 模型提供左右两个原生 `8×16` 夹爪触觉阵列，分别保存为
`sensor.tactile_left` 和 `sensor.tactile_right`。不进行尺寸插值或真机维度适配。
多模态编码器在网络输入端将左右阵列堆叠为 `[2,8,16]` 双通道张量。

策略触觉使用一条固定的信号处理管线。XML taxel 的原始编号是每个物理行
8 个、共 16 行；读取时显式转为数据约定的 `[8,16] = [物理列,物理行]`，
不会直接 reshape 后打乱空间相邻关系。

```text
每个taxel与红方块表面的精确mj_geomDistance
  → 高斯距离响应 exp(-(distance/sigma)^2)
  → 超过reach或低于threshold的响应置零
  → 按官方编号转换为原生8×16网格
  → 3x3 [1,2,1]高斯空间平滑
  → 50Hz时间EMA
  → sensor.tactile_left/right
```

`sensor.tactile_left_raw/right_raw` 仅用于诊断，不进入 ACT。数据集根目录和触觉
checkpoint 都保存 `rebot_sim_tactile_processing.json`；训练、追加采集和部署
会检查信号来源、网格布局、距离核、限幅、空间平滑、EMA 和接触时间常数与
YAML 完全一致，不一致时直接停止。仅影响界面的
`visualization_color_max` 不参与该兼容性检查。

采集和重播界面均显示处理后的 `sensor.tactile_left/right`，因此两者使用相同的
空间平滑与时间 EMA，不会把仅供诊断的逐 taxel 原始距离响应直接显示成跳动噪点。
左右热图分别显示各自 pad 的测量结果，不再把两侧镜像平均；圆柱接触因此显示为
各自实际位置上的纵向素线，不会被合成为“两边亮、中间暗”的双线假象。

该逻辑参考官方脚本 `rebotarm_colored_with_gripper_vis_rgb_8_16.py`：256 个
taxel 小方块仅用于显示和距离定位，不加入物理碰撞约束；实际夹持载荷仍由左右
手指碰撞几何承担。这样既获得连续触觉图，也避免数百个接触约束造成卡顿。

## 4. 时序定义

整个仿真流水线统一使用 50 Hz，与真机 ACT 对齐。MuJoCo 物理步长为
`0.0025 s`（400 Hz），每 8 个物理步执行一次采样、策略推理和控制更新。
一个 50 Hz 数据周期严格按以下顺序执行：

```text
推进8个MuJoCo物理步
  → 读取当前图像、关节反馈、IMU和50Hz距离触觉
  → 根据当前观测读取键盘并求解IK目标
  → 保存 (当前观测, 本周期下发目标)
  → 目标在后续物理步中执行
```

旧仿真脚本把 `env.step()` 返回的当前关节状态保存为动作，容易形成滞后一拍或
状态复制标签。新实现始终保存 `compute_q`，保证动作就是控制目标。

## 5. 使用

所有命令在仓库根目录执行。

当前 viewer 已适配 MuJoCo 3.11 的五参数 `mjv_moveCamera` 接口，鼠标左键旋转、
右键平移和滚轮缩放均可直接用于采集窗口。依赖允许使用 MuJoCo `3.1.6` 至
`4.0` 之前的 3.x 版本。

### 5.1 采集

```bash
python -m rebot_act_sim.workflow.collect --episodes 20
```

无需键盘操作的自动采集使用：

```bash
python -m rebot_act_sim.workflow.collect --auto_collect --episodes 20

# 调整整段轨迹速度（默认 1.0 倍；建议保持在 0.8～1.2）
python -m rebot_act_sim.workflow.collect --auto_collect --episodes 20 --auto-speed 0.8
```

若要排除方块棱角对夹持观感的影响，可使用独立的 4 cm 红色圆柱场景进行对照：

```bash
python -m rebot_act_sim.workflow.collect \
  --config rebot_act_sim/configs/act_sim_cylinder.yaml \
  --auto_collect --episodes 20
```

圆柱测试使用独立数据目录 `data_act_sim/rebot_act_red_cylinder`，不会与方块数据集
混合。重播时使用同一份配置：

```bash
python -m rebot_act_sim.workflow.replay \
  --config rebot_act_sim/configs/act_sim_cylinder.yaml --episode 0
```

`--auto_collect` 会在每个 episode 开始时读取本次随机化后的方块和托盘位置，
依次执行张开夹爪、移动到方块上方、下降夹持、抬升、移动到托盘中心、闭环校正、
稳定释放和退回。笛卡尔轨迹采用端点速度与加速度均为零的五次平滑插值，并在
托盘上方及释放前后增加稳定阶段；IK 关节目标另有限速与加速度约束，以减少
机械臂抖动和方块释放后的横向漂移。
保存的 `action` 是每个 50 Hz 周期实际下发的六关节 IK 目标与夹爪目标。只有任务
成功并持续满足成功判定后才保存 episode；超过 `--max-frames` 的失败尝试会丢弃、
重新随机化并自动重试。自动模式仍会显示 MuJoCo 和传感器画面，关闭窗口可安全退出。

平移键位与旧环境一致：`W/A/S/D` 控制水平移动，`R/F` 控制升降。旋转键位在
本包中重新定义为：

```text
↑ / ↓    夹爪朝前 / 朝后俯仰
← / →    从末端方向观察时逆时针 / 顺时针滚转
Q / E    左 / 右偏航
Space    切换夹爪
Z        丢弃当前episode并重置
```

这些旋转均绕官方 `end_link` 局部坐标轴执行。第一次创建的数据默认位于
`data_act_sim/rebot_act_red_cube`。

平移步长为每个 50 Hz 周期 `0.003 m`，旋转步长为 `0.02 rad`。采集端对连续
运动增量使用 `environment.teleop.motion_alpha`（默认 `0.25`）做一阶加减速，
避免键盘位置阶跃激发夹爪和接触求解器；这不会改变数据采集的 50 Hz 频率。

采集窗口左上角会实时显示当前 MuJoCo 帧的 IMU 曲线和左右 `8×16` 触觉热图，
布局与数据回放面板一致。面板由后台线程绘制；开始新 episode 时曲线历史和触觉
滤波状态会自动清零。末端绿色方向线已改为浅绿色、半透明细线，减少画面遮挡。

采集默认以 25 Hz 更新 viewer（`--render-every 2`），双相机默认按
`environment.camera_render_hz: 25` 实际渲染；数据集、关节、动作、IMU、触觉和
时间轴仍保持 50 Hz，中间控制帧复用最近的双相机图像。无用的 `sideview` 不再
渲染。可按机器性能调整：

```bash
# 更低延迟：相机有效10Hz、viewer有效25Hz
python -m rebot_act_sim.workflow.collect --episodes 20 --camera-render-hz 10

# 恢复双相机与viewer均为50Hz（GPU足够快时使用）
python -m rebot_act_sim.workflow.collect --episodes 20 \
  --camera-render-hz 50 --render-every 1
```

触觉处理参数位于同一环境配置中：

```yaml
tactile_processing:
  signal_source: distance_proximity
  normal_axis: 2
  projection_sigma: 0.0025
  clip_max: 1.0
  temporal_ema_alpha: 0.25
  spatial_smoothing: true
  contact_time_constant: 0.01
  proximity_sigma: 0.00030
  proximity_reach: 0.00125
  proximity_threshold: 0.05
  visualization_color_max: 1.0
```

`proximity_sigma` 和 `proximity_reach` 单位为米，分别控制距离响应衰减和最大
感知范围；`proximity_threshold` 抑制微弱远场响应。`visualization_color_max`
只控制显示；其余参数定义保存到数据集和触觉 checkpoint 的策略触觉语义。
`continuous_contact_projection` 与 `legacy_force_sensor` 保留用于旧数据诊断，
新 B601 数据默认使用 `distance_proximity`。

触觉处理规格已升级为 format version 3，不能向使用旧法向力语义的数据集直接
追加。重新采集时请改用新的数据集目录，或确认旧数据无需保留后执行：

```bash
python -m rebot_act_sim.workflow.collect --episodes 20 --overwrite
```

热图使用平方根对比度增强低强度区域，因此显示颜色不再与策略保存数值一一
线性对应；数据本身仍保存未经显示变换的接近强度。

物体初始化由 `environment.object_randomization` 统一控制：

```yaml
object_randomization:
  plate_position: [0.27, -0.16, 0.82]
  target_object_position_center: [0.32, 0.08, 0.8205]
  target_object_xy_half_range: [0.025, 0.025]
```

盘子在所有 episode 中使用同一个坐标，并在每个 MuJoCo 物理步后锁定自由关节
位置、单位四元数和零速度，因此不会被机械臂或红色方块推走。红色方块中心位置为
`[0.32,0.08,0.8205]`，仅在 X/Y 方向各随机 `±0.025 m`，Z 高度和初始姿态不变。
非负 seed 使用独立随机数生成器，因此同一 seed 可复现同一个方块位置。

采集和重播设置物体位姿时只执行 MuJoCo `forward`，不会在瞬移后额外执行物理
步；方块底面与桌面之间保留 `0.5 mm` 安全间隙，再由正常重力接触逐步建立支撑，
避免初始穿透导致物体弹飞。

显式覆盖已有 LeRobot 数据集：

```bash
python -m rebot_act_sim.workflow.collect --episodes 20 --overwrite
```

非负 seed 会使用 `base_seed + episode_index`，既可复现又能让不同 episode
具有不同物体布局；`--seed -1` 每次随机初始化。

### 5.2 检查与回放

```bash
python -m rebot_act_sim.workflow.inspect_dataset
python -m rebot_act_sim.workflow.replay --episode 0
```

部署推理默认在 MuJoCo 界面左上角显示 IMU 与左右 `8×16` 触觉面板，面板数据与
策略当前观测同步。需要只保留干净的仿真界面时：

```bash
python -m rebot_act_sim.workflow.deploy --no-sensors
```

部署和回放均可通过 `--render-every N` 降低 MuJoCo viewer 的渲染频率：

```bash
python -m rebot_act_sim.workflow.deploy --render-every 2
python -m rebot_act_sim.workflow.replay --episode 0 --render-every 2
```

`--render-every` 设为大于 1 的值时，每 N 个控制周期才执行一次 `env.render()`，
非渲染帧只推进物理和策略推理，跳过 MuJoCo 画面更新。默认 `1` 保持每帧渲染。
部署传感器面板默认开启，可使用 `--no-sensors` 关闭。
面板显示的是送入策略的当前 `sensor.imu`、`sensor.tactile_left/right`，不是
上一帧或数据集缓存。

MuJoCo 窗口支持物体扰动：双击物体完成选取，随后按住 `Ctrl+右键` 拖动平移，
按住 `Ctrl+左键` 拖动旋转。普通左/右键拖动仍用于调整自由相机。

检查器逐帧验证图像、shape、dtype、有限值和夹爪范围，并报告触觉全零帧数。
抓取前触觉为零是正常的；如果所有帧均为零，应检查碰撞和传感器 XML。

回放窗口左上角默认显示与当前数据帧同步的多模态面板：

- IMU 四元数当前值；
- 最近 2 秒的三轴陀螺仪曲线；
- 最近 2 秒的三轴加速度曲线；
- 左右 `8×16` 触觉热图；
- 每侧最大值、均值和压力中心；
- 当前数据帧编号和数据集时间戳。

左右热图使用固定的共享色标，确保不同帧和两侧颜色可以直接比较，不会因单点尖峰
导致整幅热图忽明忽暗。需要只查看相机和机器人
动作时可关闭传感器面板：

```bash
python -m rebot_act_sim.workflow.replay --episode 0 --no-sensors
```

回放默认按数据集 50 Hz 的真实墙钟速度执行，即每帧 20 ms。MuJoCo 仿真 tick
只负责决定何时产生控制帧，`WallClockRate` 另外负责限制 viewer 消费数据的速度，
因此回放不会因为计算量比采集小而加速。可显式调整回放倍速：

```bash
python -m rebot_act_sim.workflow.replay --episode 0 --speed 0.5  # 半速
python -m rebot_act_sim.workflow.replay --episode 0 --speed 1.0  # 实时50Hz
python -m rebot_act_sim.workflow.replay --episode 0 --speed 2.0  # 二倍速
```

采集和策略部署固定使用 50 Hz 墙钟节拍，不提供倍速选项。某帧处理超过 20 ms
时，节拍器会记录 deadline miss 并从当前时间重新建立节拍，不会通过连续快速执行
后续帧来追赶。

### 5.3 训练视觉 ACT

训练入口会先执行与真机 ACT 项目同级别的预检：校验数据集 FPS、策略字段、
episode 最短长度、未来动作窗口以及图像、关节、IMU、左右触觉 shape。只检查
而不启动训练可运行：

```bash
python -m rebot_act_sim.workflow.train --check-only --imu --tactile
```

```bash
python -m rebot_act_sim.workflow.train
```

### 5.4 训练多模态 ACT

仅 IMU：

```bash
python -m rebot_act_sim.workflow.train --imu --no-tactile
```

仅触觉：

```bash
python -m rebot_act_sim.workflow.train --no-imu --tactile
```

IMU 与触觉：

```bash
python -m rebot_act_sim.workflow.train --imu --tactile
```

也可以直接修改 `configs/act_sim.yaml` 中的 `multimodal` 开关。IMU 经过 MLP，
触觉经过保留二维结构的 CNN；编码结果拼接成一个
`observation.environment_state` token，交给 ACT Transformer。

训练循环已与验证无误的 `rebot_act_real` 逻辑对齐：默认采用 ACT 自带 AdamW
预设（学习率 `1e-5`）、梯度范数裁剪、CUDA TF32/卷积优化，并保留最后一个
不满 batch 的数据批次。`multimodal.sensor_dropout` 会在训练时随机屏蔽完整传感器
分支，避免仿真中接触状态与任务阶段强相关而形成“触觉二值开关”；
`tactile_fusion_gain` 则限制触觉 token 压过视觉和关节状态。两项参数都会保存在
checkpoint 中并由部署自动恢复。

旧多模态 checkpoint 不会因代码升级而自动获得这些训练约束。若旧权重表现为
只下沉、输出固定，或动作只随有无触觉切换，需要用新配置重新训练，而不是继续
调整旧权重的部署参数。

训练过程中按 `save_freq` 间隔保存中间 checkpoint（如 `step_002000`、`step_004000`），
训练循环结束后**始终保存最终步 checkpoint**（`checkpoints/step_{total_steps:06d}`）
以及 `pretrained_model`（eval 模式）。无论 `total_steps` 是否被 `save_freq` 整除，
最终步都不会丢失。输出目录结构示例：

```text
rebot_act_sim/ckpt/act_sim_red_cube/
├── checkpoints/
│   ├── step_002000/              # save_freq=2000 中间检查点（train 模式）
│   └── step_004000/              # 最终步检查点（train 模式，始终保存）
├── pretrained_model/             # eval 模式最终模型
├── training_metrics.json
├── loss_curve.png
└── run_config.json
```

### 5.5 部署

```bash
python -m rebot_act_sim.workflow.deploy
```

部署默认在每次启动时从配置的 X/Y 范围内重新随机物体位置，并在启动日志的
`[DEPLOY RESET]` 中输出实际坐标。需要复现实验时才显式传入非负 seed，例如
`--seed 0`；同一个 seed 始终产生相同布局，`--seed -1` 与省略参数等价。

部署会根据 checkpoint 中的 `rebot_sim_multimodal.json` 自动判断是否需要
IMU/触觉，并据此决定是否启用 50 Hz 距离触觉计算。部署控制与
`rebot_aerohand_right_act_sim` 对齐：`n_action_steps=1`，使用 ACT 原生时间集成；
当前任务默认采用实际验证可抓取的 `temporal_ensemble=0.01`，不对夹爪输出增加补偿。
相机与采集保持 `camera_render_hz=25`，两个 50 Hz 控制帧共享同一相机图像，避免
部署阶段相机时序分布变化并降低渲染负载。

```bash
python -m rebot_act_sim.workflow.deploy --temporal-ensemble 0.01
```

部署和真机项目一样从 checkpoint 自带的 `config.json` 恢复 ACT 网络、
`chunk_size` 与 `n_action_steps`，不会使用当前 YAML 重建旧权重结构。
仿真部署默认 `n_action_steps=1`，可用 `--n-action-steps` 显式修改。部署不会对
六关节或夹爪预测添加补偿、滤波或二次整形，动作按采集/replay 相同语义直接交给
`env.command()`。部署不会额外冻结松爪阶段的机械臂；否则 ACT 会持续看到静止
观测并丢失已经预测出的上升撤离动作。从接近、抓取、搬运、放置、松爪到上升撤离，
全部六关节和夹爪命令均来自 ACT 模型。指定
`--temporal-ensemble [系数]` 时会按 LeRobot ACT 要求将 `n_action_steps` 设为 1。
XML 负责场景和执行机构，不再隐式改变 checkpoint 的策略特征定义。

如需显式使用默认逐步闭环设置：

```bash
python -m rebot_act_sim.workflow.deploy \
  --n-action-steps 1 --temporal-ensemble 0.01
```

**性能优化**（为保障 50 Hz 实时性）：

| 优化项 | 机制 | 节省 |
|--------|------|------|
| 跳过 sideview 摄像机渲染 | 部署时 `grab_sideview=False`，少一次 `mj_render` 循环 | ~3-5 ms |
| 触觉按需启用 | 标准 ACT 自动跳过每物理步的接触投影；仅多模态触觉 checkpoint 启用 | ~5-8 ms |
| 传感器面板异步渲染 | 后台线程并行构造 IMU/触觉面板，与 GPU 推理重叠 | ~5-10 ms |
| Viewer 渲染节流 | `--render-every N` 降低 MuJoCo 画面更新频率 | ~5-10 ms |

```bash
# 默认：自动检测 checkpoint，跳过不必要的触觉处理
python -m rebot_act_sim.workflow.deploy

# 每两帧渲染一次 MuJoCo 画面
python -m rebot_act_sim.workflow.deploy --render-every 2

# 默认显示左上角 IMU 和左右触觉面板；如需关闭
python -m rebot_act_sim.workflow.deploy --no-sensors

# 强制禁用触觉处理（即使 checkpoint 编码了触觉分支）
python -m rebot_act_sim.workflow.deploy --no-tactile

# 指定 checkpoint
python -m rebot_act_sim.workflow.deploy --checkpoint rebot_act_sim/ckpt/act_sim_red_cube/checkpoints/step_004000
```

部署会在闭环中持续运行直到成功（方块放置到盘子上且夹爪打开、末端撤离并保持
0.5 秒）或达到 `--max-steps`（默认 1500 步，50 Hz 下约 30 秒）。结束时会输出末端/物体位移、动作
范围和最后动作。如果 `min` 与 `max` 七维都相同，说明 checkpoint 已退化为恒定
动作输出，应使用本节新版多模态训练约束重新训练，而不是继续调整 MuJoCo 控制器。

## 6. 与真机 ACT 的对应关系

| 语义 | 仿真 | 真机 |
|---|---|---|
| 状态 | MuJoCo六关节反馈 | 编码器/电机六关节反馈 |
| 动作 | 六关节绝对目标 + 0/1夹爪 | 六关节绝对目标 + 达妙夹爪目标 |
| IMU | MuJoCo末端framequat/gyro/accel | 串口IMU |
| 触觉 | 左右两个独立8×16力阵列 | FlexiTac 12×30 |
| 相机 | MuJoCo agentview/cam_wrist | Astra/RealSense |

夹爪最后一维和触觉阵列维度在仿真与真机间不同，因此 checkpoint 不能未经适配
直接跨域执行。左右触觉的字段语义保持清晰，后续如需 sim-to-real，应增加显式的
触觉适配器，而不是在仿真数据采集阶段改变原始维度。

## 7. 设计约束

- `sensor.*` 默认只是数据字段，不会被 LeRobot 自动加入标准 ACT。
- 训练和部署必须使用同一份 YAML、数据集统计量和 checkpoint。
- 改变状态、动作或触觉语义时应创建新数据集，不能向旧数据集混合追加。
- 仿真成功判定要求红色方块位于盘子附近、夹爪打开且末端已经向上撤离，并连续保持
  25 个控制周期，即 0.5 秒。保持时间由配置项
  `environment.success_hold_seconds` 控制，并按 50 Hz 自动换算为周期数。
- 离线 loss 不能替代多 seed 闭环成功率。正式实验应固定 seed 集合并统计成功率、
  完成时间和失败类型。
