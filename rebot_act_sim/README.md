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
`rebot_act_sim/assets/example_scene_rebot_red_cube.xml`。原杯子网格已经替换为
纯红色方块：

```xml
<geom type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/>
```

MuJoCo 的 box `size` 表示三个方向的半尺寸，因此方块实际边长为 `0.04 m`。
方块质量为 `0.05 kg`，具有自由关节和独立碰撞几何，可以被正常抓取和放置。

## 2. 包结构

```text
rebot_act_sim/
├── configs/act_sim.yaml          # 数据、环境、ACT和训练参数
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
sensor.tactile_left_raw            float32 [8,16]  当前物理步原始左法向力
sensor.tactile_right_raw           float32 [8,16]  当前物理步原始右法向力
sensor.sim_time                    float64 [1]
episode.object_initial_position    float32 [6]
```

`sensor.imu` 的顺序是末端 `quaternion(wxyz) + gyro(xyz) + accel(xyz)`。

MuJoCo 模型提供左右两个原生 `8×16` 夹爪力阵列，分别保存为
`sensor.tactile_left` 和 `sensor.tactile_right`。不进行尺寸插值或真机维度适配。
多模态编码器在网络输入端将左右阵列堆叠为 `[2,8,16]` 双通道张量。

策略触觉使用一条固定的信号处理管线。XML taxel 的原始编号是每个物理行
8 个、共 16 行；读取时显式转为数据约定的 `[8,16] = [物理列,物理行]`，
不会直接 reshape 后打乱空间相邻关系。

```text
每侧一个连续碰撞面上的MuJoCo接触区域与法向力
  → 接触区域平滑投影到原生8×16网格
  → 每物理步限幅
  → 8个400Hz物理步求平均
  → 3x3 [1,2,1]高斯空间平滑
  → 50Hz时间EMA
  → sensor.tactile_left/right
```

`sensor.tactile_left_raw/right_raw` 仅用于诊断，不进入 ACT。数据集根目录和触觉
checkpoint 都保存 `rebot_sim_tactile_processing.json`；训练、追加采集和部署
会检查信号来源、网格布局、投影核、限幅、空间平滑、EMA 和接触时间常数与
YAML 完全一致，不一致时直接停止。仅影响界面的
`visualization_color_max` 不参与该兼容性检查。

连续碰撞面用于避免 128 个共面小碰撞体之间出现不唯一的接触力分配。各 taxel
小方块仍保留用于显示和定位，但在本环境实例中不参与碰撞。

## 4. 时序定义

整个仿真流水线统一使用 50 Hz，与真机 ACT 对齐。MuJoCo 物理步长为
`0.0025 s`（400 Hz），每 8 个物理步执行一次采样、策略推理和控制更新。
一个 50 Hz 数据周期严格按以下顺序执行：

```text
推进8个MuJoCo物理步并累积触觉
  → 读取当前图像、关节反馈、IMU和处理后触觉
  → 根据当前观测读取键盘并求解IK目标
  → 保存 (当前观测, 本周期下发目标)
  → 目标在后续物理步中执行
```

旧仿真脚本把 `env.step()` 返回的当前关节状态保存为动作，容易形成滞后一拍或
状态复制标签。新实现始终保存 `compute_q`，保证动作就是控制目标。

## 5. 使用

所有命令在仓库根目录执行。

### 5.1 采集

```bash
python -m rebot_act_sim.workflow.collect --episodes 20
```

平移键位与旧环境一致：`W/A/S/D` 控制水平移动，`R/F` 控制升降。旋转键位在
本包中重新定义为：

```text
↑ / ↓    夹爪朝前 / 朝后俯仰
← / →    从末端方向观察时逆时针 / 顺时针滚转
Q / E    左 / 右偏航
Space    切换夹爪
Z        丢弃当前episode并重置
```

这些旋转均绕 `tcp_link` 局部坐标轴执行。第一次创建的数据默认位于
`data_act_sim/rebot_act_red_cube`。

平移步长为每个 50 Hz 周期 `0.003 m`，旋转步长为 `0.02 rad`。采集端对连续
运动增量使用 `environment.teleop.motion_alpha`（默认 `0.25`）做一阶加减速，
避免键盘位置阶跃激发夹爪和接触求解器；这不会改变数据采集的 50 Hz 频率。

采集窗口左上角会实时显示当前 MuJoCo 帧的 IMU 曲线和左右 `8×16` 触觉热图，
布局与数据回放面板一致。开始新 episode 时曲线历史和触觉滤波状态会自动清零。

可通过 `--render-every N` 控制 MuJoCo viewer 的渲染频率（默认 `1`，即每个控制周期
渲染一次）。设置为 `2` 或更大值时可降低渲染开销，提升控制循环的实时性：

触觉处理参数位于同一环境配置中：

```yaml
tactile_processing:
  signal_source: continuous_contact_projection
  normal_axis: 2
  projection_sigma: 0.0025
  clip_max: 25.0
  temporal_ema_alpha: 0.25
  spatial_smoothing: true
  contact_time_constant: 0.01
  visualization_color_max: 15.0
```

其中 `projection_sigma` 单位为米，控制接触区域边界的平滑宽度；
`visualization_color_max` 只控制显示；其余参数定义保存到数据集和触觉
checkpoint 的策略触觉语义。`legacy_force_sensor` 仅用于旧模型诊断，不建议
用于新数据采集。

热图使用平方根对比度增强低压力区域，因此显示颜色不再与策略保存数值一一
线性对应；数据本身仍保存未经显示变换的法向力。

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

部署推理默认也会在 MuJoCo 窗口左上角显示与当前策略观测同步的 IMU 曲线和左右
`8×16` 处理后触觉热图：

```bash
python -m rebot_act_sim.workflow.deploy --seed 0
```

部署和回放均可通过 `--render-every N` 降低 MuJoCo viewer 的渲染频率：

```bash
python -m rebot_act_sim.workflow.deploy --seed 0 --render-every 2
python -m rebot_act_sim.workflow.replay --episode 0 --render-every 2
```

`--render-every` 设为大于 1 的值时，每 N 个控制周期才执行一次 `env.render()`，
非渲染帧只推进物理和策略推理，跳过 MuJoCo 画面更新。默认 `1` 保持每帧渲染。
如需完全关闭传感器面板（同时关闭 MuJoCo 画面更新时除外），可添加 `--no-sensors`。
面板显示的是送入策略的当前 `sensor.imu`、`sensor.tactile_left/right`，不是
上一帧或数据集缓存。

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
python -m rebot_act_sim.workflow.deploy --seed 0
```

部署会根据 checkpoint 中的 `rebot_sim_multimodal.json` 自动判断是否需要
IMU/触觉，并据此决定是否启用每物理步的触觉接触投影。默认每个控制周期重新
预测动作块，并以系数 0.9 做时间集合。

**性能优化**（为保障 50 Hz 实时性）：

| 优化项 | 机制 | 节省 |
|--------|------|------|
| 跳过 sideview 摄像机渲染 | 部署时 `grab_sideview=False`，少一次 `mj_render` 循环 | ~3-5 ms |
| 触觉按需启用 | 标准 ACT 自动跳过每物理步的接触投影；仅多模态触觉 checkpoint 启用 | ~5-8 ms |
| 传感器面板异步渲染 | 后台线程并行构造 IMU/触觉面板，与 GPU 推理重叠 | ~5-10 ms |
| Viewer 渲染节流 | `--render-every N` 降低 MuJoCo 画面更新频率 | ~5-10 ms |

```bash
# 默认：自动检测 checkpoint，跳过不必要的触觉处理
python -m rebot_act_sim.workflow.deploy --seed 0

# 每两帧渲染一次 MuJoCo 画面
python -m rebot_act_sim.workflow.deploy --seed 0 --render-every 2

# 完全关闭传感器面板
python -m rebot_act_sim.workflow.deploy --seed 0 --no-sensors

# 强制禁用触觉处理（即使 checkpoint 编码了触觉分支）
python -m rebot_act_sim.workflow.deploy --seed 0 --no-tactile

# 指定 checkpoint
python -m rebot_act_sim.workflow.deploy --checkpoint rebot_act_sim/ckpt/act_sim_red_cube/checkpoints/step_004000
```

部署会在闭环中持续运行直到成功（方块放置到盘子上且夹爪打开、末端撤离并保持
0.5 秒）或达到 `--max-steps`（默认 800 步）。

## 6. 与真机 ACT 的对应关系

| 语义 | 仿真 | 真机 |
|---|---|---|
| 状态 | MuJoCo六关节反馈 | 编码器/电机六关节反馈 |
| 动作 | 六关节绝对目标 + 0/1夹爪 | 六关节绝对目标 + 达妙夹爪目标 |
| IMU | MuJoCo末端framequat/gyro/accel | 串口IMU |
| 触觉 | 左右两个独立8×16力阵列 | FlexiTac 12×30 |
| 相机 | MuJoCo agent/egocentric | Astra/RealSense |

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
