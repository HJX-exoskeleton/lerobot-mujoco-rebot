# reBot + AeroHand 右手 ACT MuJoCo 仿真工程

## 1. 目标

`rebot_aerohand_right_act_sim` 是 6 自由度 reBot 机械臂 + 右手 AeroHand 灵巧手的
规范化 ACT 仿真流水线，工程结构与 `rebot_act_sim` 对齐，并复用同一套工程原则：

- 采集、训练和部署共享一份数据契约；
- 明确区分机器人反馈状态与下发动作目标；
- 标准 ACT 使用双相机（top 俯视相机 + cam_wrist 腕部相机）与机械臂关节反馈；
- 灵巧手 16 关节与 7 执行器的反馈作为辅助字段永久保存在数据集中；
- IMU、手部接触力作为辅助字段永久保存在数据集中；
- 只有显式启用多模态训练时，辅助传感器才进入网络；
- checkpoint 保存自身的多模态规格，部署时自动识别；
- 仿真代码不依赖任何真机驱动，也不依赖 MediaPipe 摄像头；
- 手部控制在数据采集中使用 MuJoCo 窗口键盘完成，不与
  `rebot_aerohand_right_control_sim` 的 CV 控制耦合。

当前任务使用专用场景
`asset_rebot_aerohand_right/mujoco_xml/rebotarm_aerohand_act_cylinder.xml`。
该场景 include 同目录下的 `rebotarm_aerohand_scene.xml`、
`rebot_arm_right_hand.xml`，并额外在 `tetheria_mount`
上挂载三个 IMU 传感器。场景必须与该目录中的其它 XML 同目录存放：MuJoCo
仅当顶层 XML 与组合模型嵌套 include 的
`aerohand_right_body.xml` 及全部网格/纹理位于同一目录时才能正确解析相对
路径（详见 `rebot_aerohand_right_act_sim/assets/README.md`）。

任务物体是桌面上的红色圆柱（`red_box`，MuJoCo cylinder `size` 表示
半径 0.02 m 与半高 0.07 m），质量为 0.075 kg，具有自由关节；目标区域是桌面
另一侧半径 0.10 m 的固定灰色圆盘（`target_box`）。目标圆盘的碰撞在本场景
XML 中直接以 `contype/conaffinity=1` 开启（不是运行时修改）：mujoco 3.11
忽略运行时对 `geom_contype`/`geom_conaffinity` 的修改，若只在代码里开启，
圆盘会变成"幽灵"物体，圆柱会直接穿过它落在桌面上，成功判定的接触条件
永远无法满足。

## 2. 包结构

```text
rebot_aerohand_right_act_sim/
├── assets/README.md                 # 任务场景位置说明（场景在资产包 mujoco_xml/ 中）
├── configs/aerohand_act_sim.yaml    # 数据、环境、ACT和训练参数
├── config.py                        # 配置与ACT特征构造
├── schema.py                        # 唯一数据契约
├── dataset_writer.py                # 安全创建、追加、保存和丢弃episode
├── environment.py                   # MuJoCo任务适配器、键盘遥操作与手部映射
├── sensors.py                       # IMU与手部接触力读取
├── multimodal_policy.py             # IMU MLP、接触力MLP和传感器token融合
├── policy.py                        # 训练/部署统一的策略构造
├── visualization.py                 # 同步/异步IMU曲线与接触力面板渲染
├── timing.py                        # WallClockRate 实时墙钟节拍器
├── workflow/
│   ├── collect.py
│   ├── replay.py
│   ├── inspect_dataset.py
│   ├── train.py
│   └── deploy.py
└── tests/
```

`environment.py` 内置三个自包含组件（不依赖 `mujoco_env.SimpleEnv`，也不依赖
`rebot_aerohand_right_control_sim`）：

- GLFW MuJoCo 窗口（自由相机 + 按键回调 + 传感器面板叠加）；
- 机械臂笛卡尔 IK 控制器（世界系平移 + 末端系旋转，`mj_jacBody` 阻尼最小二乘）；
- AeroHand 执行器映射（5 根手指的二元开合状态线性映射到 7 个执行器目标）。

## 3. 数据契约

策略基础字段：

```text
observation.image        uint8   [256,256,3]  top 俯视相机
observation.wrist_image  uint8   [256,256,3]  cam_wrist 腕部相机
observation.state        float32 [6]          机械臂六关节反馈，rad
action                   float32 [13]         六关节绝对目标rad + 7维灵巧手执行器绝对目标
```

`action` 后 7 维是 AeroHand 七个执行器的绝对目标，顺序与
`aerohand_right.xml` 的 actuator 声明一致：

```text
[0:4] 食指/中指/无名指/小指屈肌腱长度 (m)，ctrlrange [0.058520, 0.110387]，大=张开
[4]   拇指外展角 (rad)，ctrlrange [-0.1, 1.75]，0=张开
[5]   拇指近端肌腱长度 (m)，ctrlrange [0.026152, 0.038389]，大=张开
[6]   拇指远端肌腱长度 (m)，ctrlrange [0.081568, 0.112138]，大=张开
```

辅助字段：

```text
sensor.joint_velocity              float32 [6]   机械臂六关节速度，rad/s
sensor.hand_feedback               float32 [7]   四指肌腱长度+拇指外展角+两拇指肌腱长度
sensor.hand_joint_position         float32 [16]  灵巧手16关节反馈，rad
sensor.imu                         float32 [10]  腕部 quaternion(wxyz)+gyro(xyz)+accel(xyz)
sensor.hand_contact                float32 [6]   拇指/食指/中指/无名指/小指/手掌法向力，N
sensor.sim_time                    float64 [1]
episode.object_initial_position    float32 [7]   圆柱初始 qpos(xyz + 单位四元数)
```

`sensor.hand_joint_position` 的顺序与模型关节声明一致：食指
mcp/pip/dip、中指、无名指、小指、拇指 cmc_abd/cmc_flex/mcp/ip。由于 XML 中的
equality 约束，pip 与 dip 关节始终相等。

`sensor.imu` 的顺序是末端 `quaternion(wxyz) + gyro(xyz) + accel(xyz)`，
由挂载在 `tetheria_mount` 的 `imu_hand_site` 上的
`framequat`/`gyro`/`accelerometer` 传感器读取，传感器名与 `rebot_act_sim`
保持一致（`orientation_left`/`ang_vel_left`/`accel_left`）。

`sensor.hand_contact` 是手部六区域接触法向力，使用一条固定的信号处理管线。
接触对中属于手部 body 树（`palm` 及其后代）的 geom 按 body 名归入六区域，
每物理步按区域累加法向力并限幅，每个 50 Hz 控制周期求平均后再做时间 EMA：

```text
接触对中手部geom与外界geom的法向力
  → 按 body 名归入 thumb/index/middle/ring/pinky/palm 六区域
  → 每物理步限幅
  → 20个1000Hz物理步按区域求平均
  → 50Hz时间EMA
  → sensor.hand_contact
```

手部接触处理参数位于环境配置中：

```yaml
hand_contact_processing:
  regions: [thumb, index, middle, ring, pinky, palm]
  clip_max: 25.0
  temporal_ema_alpha: 0.25
  visualization_color_max: 15.0
```

数据集根目录和接触力 checkpoint 都保存
`rebot_aerohand_right_hand_contact_processing.json`；训练、追加采集和部署会检查
区域定义、限幅与 EMA 参数与 YAML 完全一致，不一致时直接停止。仅影响界面的
`visualization_color_max` 不参与该兼容性检查。

## 4. 时序定义

整个仿真流水线统一使用 50 Hz，与真机 ACT 对齐。组合模型的物理步长为
`0.001 s`（1000 Hz，由 `rebor_arm_6dof.xml` 定义），每 20 个物理步执行一次
采样、策略推理和控制更新。一个 50 Hz 数据周期严格按以下顺序执行：

```text
推进20个MuJoCo物理步（每步写入手部滤波目标并累积接触力）
  → 读取当前图像、关节反馈、IMU和处理后接触力
  → 根据当前观测读取键盘、更新IK目标与手部目标
  → 保存 (当前观测, 本周期下发目标)
  → 目标在后续物理步中执行
```

机械臂 IK 解算出的 `command_q` 就是保存的动作前 6 维；手部保存的同样是本周期
下发的 7 维执行器目标，而不是物理步内的滤波中间值。手部目标在物理步内以
`hand_command_alpha`（默认 `0.25`，与 `rebot_aerohand_right_control_sim`
一致）做一阶滤波后写入 `data.ctrl`，避免阶跃目标激发肌腱与接触求解器；
这不会改变数据采集的 50 Hz 频率，也不改变保存的动作语义。

双相机默认按 `environment.camera_render_hz: 25` 渲染，50 Hz 控制序列中的中间
帧复用最近一次 `top` 与 `cam_wrist` 图像。机械臂状态、动作、IMU、接触力和
数据集时间轴仍保持 50 Hz；只有视觉内容的有效更新率为 25 Hz。该同步缓存方案
不跨线程共享 `MjData` 或 OpenGL context，避免异步渲染导致图像与状态标签错位。

## 5. 使用

所有命令在仓库根目录执行。

### 5.1 采集

```bash
python -m rebot_aerohand_right_act_sim.workflow.collect --episodes 20
```

自动采集使用：

```bash
python -m rebot_aerohand_right_act_sim.workflow.collect \
  --auto_collect \
  --episodes 20
```

如果相机渲染仍然拖慢仿真，可进一步降低有效图像帧率：

```bash
python -m rebot_aerohand_right_act_sim.workflow.collect \
  --auto_collect --camera-render-hz 10 --episodes 20
```

`--camera-render-hz` 只改变相机实际渲染频率，不改变数据集、状态和 action 的
50 Hz 帧率；未重新渲染的控制帧会保存最近的有效图像。

`--auto_collect` 会为每个随机化后的圆柱自动执行完整示教：张开手稳定、移动到
预抓取位、下降、拇指虎口对掌、向前接近、五指包络、稳定抓取、抬升、移动到
目标盘、下降、松抓、释放和撤离。动作经过现有 IK 与手部目标滤波器，并按普通
采集完全相同的 observation/action 数据契约逐帧保存。

自动轨迹默认使用 `2.5x` 速度，通常约 539 帧完成，低于默认的 800 帧上限。
可使用 `--auto-speed` 调整，例如：

```bash
python -m rebot_aerohand_right_act_sim.workflow.collect \
  --auto_collect --auto-speed 2.0 --episodes 20
```

自动模式仍显示 MuJoCo 窗口和相机 PIP，但忽略键盘遥操作；关闭窗口可提前退出。
机械臂相邻 waypoint 使用连续线性笛卡尔插值，不会在每个阶段边界强制降速到
零；抓取稳定、虎口预成形和释放等明确阶段仍按设计保持。采集界面的 PIP 直接
复用写入数据集的 `top` 与 `cam_wrist` 图像，不再重复离屏渲染，降低界面卡顿。
未满足放置、释放和撤离成功条件的 episode 仍会按原逻辑在达到
`--max-frames` 后丢弃并随机重置。

机械臂平移键位与 `rebot_act_sim` 一致：`W/A/S/D` 控制水平移动，`R/F` 控制
升降。旋转均绕 `tetheria_mount` 局部坐标轴执行：

```text
↑ / ↓    末端朝前 / 朝后俯仰
← / →    从末端方向观察时逆时针 / 顺时针滚转
Q / E    左 / 右偏航
```

灵巧手键位：

```text
1 / 2 / 3 / 4 / 5   切换 拇指/食指/中指/无名指/小指 的开合
Space               整体抓握/张开切换
O / C               全部张开 / 分阶段闭合
H                   机械臂回初始位形并张开手
Z                   丢弃当前episode并重置
```

手指开合是二元状态：张开时执行器目标为 `ctrlrange` 上界（拇指外展为 0），
闭合时为下界（拇指外展为 1.5），与 `SimHandMapper` 的 open/closed 语义一致。
按 `C` 或用 `Space` 从张开切换到抓握时，控制分为两阶段：先只旋转拇指
CMC/虎口对掌轴，等待 `environment.teleop.thumb_lead_seconds`（默认 `0.12 s`），
再闭合拇指两路屈肌腱与其余四指。按 `O`、`H` 或单独切换手指会取消尚未完成
的分阶段闭合。

每次环境复位都会显式清零 16 个手部关节并下发张开目标。正式检测到第一条
机械臂或手部动作前，采集器还维护 `0.20 s`（50 Hz 下 10 帧）的张开状态滚动
缓冲；录制启动时先写入这些帧，保证每个 episode 的起始观测和 action 都包含
明确的张开手状态，而不是直接从闭合命令开始。
第一次创建的数据默认位于 `data_aerohand_act_sim/rebot_act_cylinder`。

平移步长为每个 50 Hz 周期 `0.003 m`，旋转步长为 `0.02 rad`。采集端对机械臂
连续运动增量使用 `environment.teleop.motion_alpha`（默认 `0.25`）做一阶
加减速，避免键盘位置阶跃激发接触求解器。

采集过程提供两层状态提示，风格与 `rebot_act_sim` 一致：

- **终端**：启动时打印场景、数据集、任务参数和完整键位说明；手指开合切换时
  打印 `[Hand] thumb:open index:closed ...`；开始录制时打印 episode 序号、
  seed 与物体初始位置；成功保存时打印
  `Saved episode i/N, frames=..., duration=...s`；按 `Z` 丢弃或达到
  `--max-frames` 时打印丢弃原因与帧数；运行中每 2 秒输出一行
  `[时间] episode=.. REC/IDLE frames=.. dist_to_obj=..`，结束后打印
  `[Summary]` 总集数与总帧数。录制超过 200 帧仍未成功时，每 200 帧打印一次
  成功条件诊断 `[HINT] placed=yes/no released=yes/no retreated=yes/no
  hold=n/25`，方便定位是哪个条件没有满足。
- **MuJoCo 窗口**：右上角叠加当前采集状态（`COLLECT episode=.. REC/IDLE`、
  帧数、仿真时间、抓取点到物体的距离、五根手指开合状态）；左下角显示
  `Key Pressed/Repeated` 与当前按键（对应 `rebot_act_sim` 的按键文本叠加）。
  回放与部署窗口同样在右上角显示各自的状态行。

窗口右侧还以画中画方式叠加相机画面（对应 `rebot_act_sim` 的 Agent/
Egocentric PIP 视图）。叠加的相机列表由配置项 `environment.camera_overlays`
控制（默认 `[top, cam_wrist]`：与送入数据集的 `observation.image` /
`observation.wrist_image` 是同一组相机，PIP 各 320×240，自上而下排列在
右上角，用于实时观察策略实际看到的两路画面；相机名标注在右上角状态行
`PIP[...]` 中）。该叠加与传感器面板同开关：回放/部署的 `--no-sensors`
会同时关闭相机叠加；置空列表可完全禁用。

采集窗口**不显示** IMU/接触力传感器面板（该面板保留在回放与部署中），
窗口内只叠加：右上角状态行、左下角按键提示、以及右上角自上而下排列的
相机 PIP 画面（top 俯视 + cam_wrist 腕部，与送入数据集的两路画面一致）。

3D 场景中会在 `aerohand_right_body.xml` 的 `hand_grasp_site` 位置绘制一个
透明度 0.2 的圆球，并用一条细线连接到红色圆柱中心，用于人工判断手掌锚点与
物体的距离。圆球颜色按抓取点到圆柱表面的距离分三档：

```text
≤ 0.06 m  绿色  已到抓取距离
≤ 0.12 m  橙色  接近中
> 0.12 m  红色  距离较远
```

该标记沿用 `rebot_act_sim` 旧 viewer 的运行时球/胶囊标记风格（`mjv_initGeom`/
`mjv_makeConnector` 逐帧注入 viewer 场景），仅存在于画面中，不进入数据、不
参与碰撞、不影响策略。若不需要，可在构造环境时设置
`show_grasp_marker=False`。

可通过 `--render-every N` 控制 MuJoCo viewer 的渲染频率（默认 `1`，即每个
控制周期渲染一次）。设置为 `2` 或更大值时可降低渲染开销，提升控制循环的
实时性。

物体初始化由 `environment.object_randomization` 统一控制：

```yaml
object_randomization:
  target_object_position_center: [0.30, -0.10, 0.07]
  target_object_xy_half_range: [0.03, 0.03]
```

红色圆柱中心位置为 `[0.30, -0.10, 0.07]`，仅在 X/Y 方向各随机 `±0.03 m`，
Z 高度和初始姿态（单位四元数）不变。目标圆盘固定在 `[0.30, 0.20]`，不参与
随机化。非负 seed 使用独立随机数生成器，因此同一 seed 可复现同一个圆柱位置。

采集和重播设置圆柱位姿时只执行 MuJoCo `forward`，不会在瞬移后额外执行物理步；
圆柱底面与桌面之间保留 `0.5 mm` 安全间隙，再由正常重力接触逐步建立支撑，
避免初始穿透导致物体弹飞。

显式覆盖已有 LeRobot 数据集：

```bash
python -m rebot_aerohand_right_act_sim.workflow.collect --episodes 20 --overwrite
```

非负 seed 会使用 `base_seed + episode_index`，既可复现又能让不同 episode
具有不同物体布局；`--seed -1` 每次随机初始化。**被丢弃的重置始终使用新的
随机位置**：按 `Z` 丢弃或达到 `--max-frames` 后的重置不再沿用原 seed，
而是重新随机化圆柱位置（保存成功的 episode 才按 `base_seed + episode_index`
确定布局，保证数据可复现）。

### 5.2 检查与回放

```bash
python -m rebot_aerohand_right_act_sim.workflow.inspect_dataset
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0
```

部署和回放均可通过 `--render-every N` 降低 MuJoCo viewer 的渲染频率：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --render-every 2
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0 --render-every 2
```

`--render-every` 设为大于 1 的值时，每 N 个控制周期才执行一次 viewer 画面
更新，非渲染帧只推进物理和策略推理。默认 `1` 保持每帧渲染。如需完全关闭
传感器面板，可添加 `--no-sensors`。面板显示的是送入策略的当前
`sensor.imu`、`sensor.hand_contact`，不是上一帧或数据集缓存。

检查器逐帧验证图像、shape、dtype、有限值和手部执行器范围，并报告接触力全零
帧数。抓取前接触力为零是正常的；如果所有帧均为零，应检查碰撞和接触分类。

回放窗口左上角默认显示与当前数据帧同步的多模态面板：

- IMU 四元数当前值；
- 最近 2 秒的三轴陀螺仪曲线；
- 最近 2 秒的三轴加速度曲线；
- 手部六区域接触力柱状图；
- 每区域当前力值与总接触力；
- 四指肌腱长度与拇指外展角；
- 当前数据帧编号和数据集时间戳。

需要只查看 3D 场景时可关闭传感器面板：

```bash
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0 --no-sensors
```

回放默认按数据集 50 Hz 的真实墙钟速度执行，即每帧 20 ms。MuJoCo 仿真 tick
只负责决定何时产生控制帧，`WallClockRate` 另外负责限制 viewer 消费数据的速度。
可显式调整回放倍速：

```bash
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0 --speed 0.5  # 半速
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0 --speed 1.0  # 实时50Hz
python -m rebot_aerohand_right_act_sim.workflow.replay --episode 0 --speed 2.0  # 二倍速
```

采集和策略部署固定使用 50 Hz 墙钟节拍，不提供倍速选项。某帧处理超过 20 ms
时，节拍器会记录 deadline miss 并从当前时间重新建立节拍，不会通过连续快速执行
后续帧来追赶。

### 5.3 训练视觉 ACT

```bash
python -m rebot_aerohand_right_act_sim.workflow.train
```

### 5.4 训练多模态 ACT

仅 IMU：

```bash
python -m rebot_aerohand_right_act_sim.workflow.train --imu --no-hand-contact
```

仅手部接触力：

```bash
python -m rebot_aerohand_right_act_sim.workflow.train --no-imu --hand-contact
```

IMU 与接触力：

```bash
python -m rebot_aerohand_right_act_sim.workflow.train --imu --hand-contact
```

也可以直接修改 `configs/aerohand_act_sim.yaml` 中的 `multimodal` 开关。IMU 与
手部接触力都经过 MLP，编码结果拼接成一个 `observation.environment_state`
token，交给 ACT Transformer。

训练过程中按 `save_freq` 间隔保存中间 checkpoint（如 `step_002000`、
`step_004000`），训练循环结束后**始终保存最终步 checkpoint**
（`checkpoints/step_{total_steps:06d}`）、`pretrained_model`（eval 模式最终
模型），并额外保存 **`best_model`**（训练中 100 步平均 loss 最低时的权重，
eval 模式）。无论 `total_steps` 是否被 `save_freq` 整除，最终步都不会丢失。
小数据集上最终步常常过拟合，部署应优先尝试 `best_model`（训练结束时会打印
`Best avg100 loss ... at step ...`）。输出目录结构示例：

```text
rebot_aerohand_right_act_sim/ckpt/act_sim_aerohand_cylinder/
├── checkpoints/
│   ├── step_002000/              # save_freq=2000 中间检查点（train 模式）
│   └── step_004000/              # 最终步检查点（train 模式，始终保存）
├── pretrained_model/             # eval 模式最终模型
├── best_model/                   # eval 模式最优100步平均loss模型
├── training_metrics.json         # 含 best_avg100_step
├── loss_curve.png
└── run_config.json
```

### 5.5 部署

部署默认执行 **50 次**随机化评估。每轮都会重置环境、策略缓存、手部平滑状态和
可视化历史，并重新随机化圆柱位置。非负 seed 与数据采集规则一致，第 `i` 轮使用
`base_seed + i`；`--seed -1` 则让每轮使用不可复现的随机位置。

默认 50 次评估：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy \
  --checkpoint rebot_aerohand_right_act_sim/ckpt/act_sim_aerohand_cylinder/pretrained_model \
  --num-rollouts 50 \
  --seed 0
```

`--num-rollouts` 的别名是 `--inference-count`。如需单次推理，可执行：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy --num-rollouts 1 --seed 0
```

单轮超过 `--max-steps`（默认 800）仍未满足成功判定时计为失败，随后自动进入
下一轮。所有轮次完成后输出成功数、失败数、成功率和成功轮次的平均策略步数。

每次部署默认在指定权重 `pretrained_model` 的上一级目录保存评估结果：

```text
<checkpoint parent>/deploy_evaluations/<timestamp>/
├── summary.json
├── rollouts.csv
├── rollout_001/
│   ├── cameras.mp4
│   ├── arm_joints.png
│   ├── hand_actuators.png
│   ├── hand_joints.png
│   ├── imu.png
│   ├── hand_contact_force.png
│   ├── object_motion.png
│   ├── tool_motion.png
│   ├── grasp_distance.png
│   └── result.json
└── rollout_002/ ...
```

`cameras.mp4` 左右拼接保存顶部相机和手腕相机。推理过程不保存原始 NPZ，而是将
机械臂实测关节/策略目标、手部执行器、手部关节、IMU、接触力、物体运动、末端
运动和抓取距离分别绘制成上述 PNG 曲线图。`result.json` 保存单轮结果和曲线图
文件列表，根目录的 JSON/CSV 汇总所有轮次和最终成功率。

自定义输出目录：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy \
  --checkpoint rebot_aerohand_right_act_sim/ckpt/act_sim_aerohand_cylinder/pretrained_model \
  --num-rollouts 50 \
  --seed 0 \
  --output-dir outputs/evaluation/run_001
```

只保存曲线图和汇总、不编码视频：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy --no-save-video
```

部署读取 `environment.camera_render_hz`，并复用策略观测图像绘制画中画，避免 viewer
为同一控制帧重复渲染相机。

部署会根据 checkpoint 中的 `rebot_aerohand_right_multimodal.json` 自动判断
是否需要 IMU/手部接触力，并据此决定是否启用每物理步的接触力累积。

**时间集合（temporal ensembling）**：部署默认每个控制周期重新预测整个动作
块，并通过 lerobot 的 `ACTTemporalEnsembler` 对历史预测做指数加权平均
（ACT 论文 Algorithm 2 的在线实现）。k 步之前的预测权重为
`exp(-coeff·k)`（k=0 即最新预测权重为 1），集合状态在整个 rollout 内持续
累积，`policy.reset()` 在 rollout 开始时清零。加权对全部 13 维动作统一
生效（机械臂 6 维 + 手部 7 维同等平滑）。系数由 `--temporal-ensemble`
指定（默认 `0.9`），checkpoint 中保存的系数在部署时被该参数覆盖：

```bash
# 默认 0.9：轻度平滑，紧跟最新预测（臂响应快，手部可能残留抖动）
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0

# 0.3：中等平滑，历史预测影响更大
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --temporal-ensemble 0.3

# 0.01：原版 ACT 默认值，强平滑（动作最稳，但整体有延迟、臂变"肉"）
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --temporal-ensemble 0.01
```

注意：**系数越小 = 平滑越强**（历史权重衰减越慢），但延迟也越大。若只想
抑制手部抖动而不拖慢机械臂，优先使用下面"手部动作稳定化"的
`--hand-smooth`/`--snap-hand`（只作用于手部 7 维），而不是调小时间集合
系数。

**性能优化**（为保障 50 Hz 实时性）：

| 优化项 | 机制 | 节省 |
|--------|------|------|
| 双相机最小渲染 | 每个控制周期只渲染 top 与 cam_wrist 两个 256×256 相机 | - |
| 接触力按需启用 | 标准 ACT 自动跳过每物理步的接触分类；仅多模态接触力 checkpoint 启用 | ~1-3 ms |
| 传感器面板异步渲染 | 后台线程并行构造 IMU/接触力面板，与 GPU 推理重叠 | ~5-10 ms |
| Viewer 渲染节流 | `--render-every N` 降低 MuJoCo 画面更新频率 | ~5-10 ms |

```bash
# 默认：自动检测 checkpoint，跳过不必要的接触力处理
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0

# 每两帧渲染一次 MuJoCo 画面
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --render-every 2

# 完全关闭左上角传感器面板与右侧相机 PIP
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --no-sensors

# 强制禁用接触力处理（即使 checkpoint 编码了接触力分支）
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --no-hand-contact

# 指定 checkpoint
python -m rebot_aerohand_right_act_sim.workflow.deploy --checkpoint rebot_aerohand_right_act_sim/ckpt/act_sim_aerohand_cylinder/checkpoints/step_004000
```

手部动作稳定化（小数据集上策略可能在闭合/张开两态间抖动，导致抓取时撞击
圆柱后抓空）：

```bash
# 时间平滑：对 7 维手部目标做 50Hz EMA（0.4~0.6 推荐）
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --hand-smooth 0.5

# 开/闭二值吸附（带滞回）：手部目标吸附到开/闭极端值，单帧噪声无法翻转状态
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --snap-hand

# 两者组合（最稳）
python -m rebot_aerohand_right_act_sim.workflow.deploy --seed 0 --hand-smooth 0.5 --snap-hand
```

这两个开关只作用于部署端的手部目标后处理，不改变数据集与训练；机械臂 6 维
动作保持策略原始输出。

**推荐参数组合**（小数据集上已验证，可稳定完成抓取任务）：

```bash
python -m rebot_aerohand_right_act_sim.workflow.deploy \
  --seed 0 --temporal-ensemble 0.1 --hand-smooth 0.5 --no-sensors
```

该组合实测推理效果好：强时间集合（0.1）+ 手部 EMA（0.5）使灵巧手动作不再
突变，闭合后不会再张开（默认 0.9 时间集合 + 无手部平滑时，策略可能在
闭合/张开两态间抖动，抓取时撞击圆柱后抓空）。`--no-sensors` 关闭传感器面板
与相机 PIP，窗口只保留 3D 场景、抓取圆球标记与状态行。注意 0.05 的强平滑
对机械臂也有整体延迟，数据量补足（10+ episodes）后建议重新对比
`--temporal-ensemble 0.3` 或默认 0.9 是否已足够稳定，再决定是否放开。

每轮部署会在闭环中持续运行，直到成功（圆柱与目标圆盘产生接触、手部张开、末端
撤离并保持 0.5 秒）或达到 `--max-steps`；之后保存该轮结果并自动进入下一轮。

## 6. 与真机 ACT 的对应关系

| 语义 | 仿真 | 真机 |
|---|---|---|
| 状态 | MuJoCo六关节反馈 | 编码器/电机六关节反馈 |
| 动作 | 六关节绝对目标 + 7执行器目标 | 六关节绝对目标 + AeroHand 7执行器目标 |
| 手部反馈 | 肌腱长度与拇指外展角 | 执行器位置反馈 |
| 手部关节 | 16关节反馈（辅助字段） | 关节编码器（可选） |
| IMU | MuJoCo末端framequat/gyro/accel | 串口IMU |
| 接触 | 六区域法向力 | 灵巧手指端力传感器（可选扩展） |
| 相机 | MuJoCo top/cam_wrist | 俯视相机/腕部相机（真机按位姿对应） |

仿真手部执行器为肌腱长度目标，真机 AeroHand 控制语义（如电机位置/电流环）需
按实机 SDK 换算，因此 checkpoint 不能未经适配直接跨域执行。后续如需
sim-to-real，应增加显式的手部控制适配器，而不是在仿真数据采集阶段改变动作
语义。

## 7. 设计约束

- `sensor.*` 默认只是数据字段，不会被 LeRobot 自动加入标准 ACT。
- 训练和部署必须使用同一份 YAML、数据集统计量和 checkpoint。
- 改变状态、动作或接触力语义时应创建新数据集，不能向旧数据集混合追加。
- 仿真成功判定要求圆柱与目标圆盘产生 MuJoCo 接触、手部四指肌腱长度接近
  张开值且末端已经向上撤离，并连续保持 25 个控制周期，即 0.5 秒。保持时间由
  配置项 `environment.success_hold_seconds` 控制，并按 50 Hz 自动换算为周期数。
- 采集数据时动作标签始终是控制目标：机械臂为 IK 解算的关节目标，手部为
  下发的执行器目标，不使用物理步内的滤波值或滞后状态。
- 离线 loss 不能替代多 seed 闭环成功率。正式实验应固定 seed 集合并统计
  成功率、完成时间和失败类型。
