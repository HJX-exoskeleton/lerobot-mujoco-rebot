# reBot ACT 真机项目开发说明

## 1. 项目目标

`rebot_act_real` 用于研究ACT（Action Chunking Transformer）在reBot六轴机械臂
香蕉抓取放置任务上的真机训练与部署。项目计划覆盖：

1. 50Hz双相机真机示教采集；
2. LeRobot数据集检查、回放和统计；
3. ACT模型训练、离线评估和checkpoint管理；
4. shadow推理、真机部署、时间集合与安全执行；
5. 后续可选的IMU、触觉和多任务扩展。

首版只实现标准视觉ACT，不加入语言、阻抗控制、轨迹引导或触觉网络。先建立一条
可复现的视觉ACT基线，再逐项扩展。

## 2. 现有代码参考关系

### 2.1 ACT仿真主线

根目录下的脚本提供了标准LeRobot ACT流程：

| 环节 | 参考代码 | 可复用内容 | 真机项目需要修改的内容 |
|---|---|---|---|
| 仿真采集 | `scripts/01_collect_data.py` | LeRobot episode生命周期、图像和动作字段 | MuJoCo遥操作改为主从真机控制 |
| 训练 | `scripts/03_train_act.py` | `ACTConfig`、数据加载、训练、误差曲线 | 参数配置化、保留双相机、增加预检和checkpoint |
| 仿真部署 | `scripts/04_deploy_act.py` | `ACTPolicy.from_pretrained`、`select_action` | MuJoCo环境改为真机观测和安全执行 |
| 仿真环境 | `mujoco_env/`、`asset_rebot/` | 关节和动作语义参考 | 不作为真机运行依赖 |

仿真训练脚本当前会移除 `observation.wrist_image`，只保留一个视觉输入。真机ACT首版
计划保留 `observation.image` 和 `observation.wrist_image` 两路相机，不能直接照搬
这项删除操作。

### 2.2 SmolVLA真机主线

`rebot_smolvla_real` 已验证的真机能力将作为实现参考：

- 50Hz主从机械臂控制与采样；
- Astra S场景相机和RealSense腕部相机；
- 六关节反馈、七维绝对动作和达妙夹爪；
- 主手舵机到真机关节的标定与映射；
- 非阻塞相机、传感器读取和异步LeRobot写入；
- episode保存、丢弃、续采和数据完整性检查；
- `SafetyGuard`、速度/单步限幅、跟踪误差保护；
- shadow模式、真机确认、可视化和安全退出；
- 已在真机验证有效的关节命令加速度连续整形。

ACT项目应移植其中通用的硬件和安全逻辑，但不能让ACT代码长期依赖
`rebot_smolvla_real` 包。首版开发可以先复制并精简已验证实现，稳定后再考虑抽取
共享的 `rebot_real_common` 模块。

### 2.3 旧ACT真机/触觉代码

`rebot_scripts/Servo_control/` 下已有较重的ACT真机和触觉实现，可用于核对：

- ACT checkpoint的旧格式；
- 时间集合实现；
- IMU和触觉输入方式；
- 真机可视化与干预控制。

它不作为首版代码骨架，因为其中混合了旧路径、旧ACT实现、触觉网络和阻抗控制。
首版优先使用当前环境安装的 `lerobot.common.policies.act.ACTPolicy`。

## 3. 首版系统边界

### 3.1 频率

整个首版工程统一为50Hz：

```text
真机控制频率       50Hz
LeRobot数据集fps   50Hz
策略输出执行频率   50Hz
单控制周期         20ms
```

相机硬件帧率不一定等于50Hz。采集程序在每个数据时刻读取两路相机的最新有效帧，
记录帧号和时间戳，并检查图像陈旧程度与双相机时间差，不伪造历史图像。

### 3.2 ACT首版策略输入和输出

策略输入：

```text
observation.image        uint8 RGB [256, 256, 3]  场景相机
observation.wrist_image  uint8 RGB [256, 256, 3]  腕部相机
observation.state        float32 [6]               真机六关节反馈位置
```

策略输出：

```text
action                   float32 [7]
action[0:6]              六关节绝对目标位置，单位rad
action[6]                夹爪绝对目标位置，单位rad
```

必须保证采集、训练、离线评估和部署始终使用同一套关节顺序、符号、单位和夹爪范围。
`observation.state` 记录真机反馈，不记录主手目标；`action` 记录经过映射、平滑和
安全限幅后实际下发的目标。

### 3.3 辅助字段

为了便于诊断，首版数据集可以继续保存：

```text
sensor.joint_velocity
sensor.gripper_feedback
sensor.frame_ids
sensor.timestamps
sensor.imu
sensor.tactile
```

这些字段默认不进入ACT网络。ACT首版的策略字段集合必须严格等于：

```text
observation.image
observation.wrist_image
observation.state
action
```

### 3.4 任务文本

LeRobot episode保存任务标签 `rebot_real_banana`，用于数据管理。
标准ACT不是语言条件策略，首版模型不会把任务文本作为网络输入。因此首版应使用
单一任务、单一动作语义的数据集，不应把多个语言任务混合后期待ACT自行区分。

## 4. 数据集规划

建议数据集位置：

```text
data_act_real/
└── rebot_act_banana/
    ├── data/
    ├── videos/
    └── meta/
```

建议标识：

```text
repo_id: rebot_real/rebot_act_banana
root: ./data_act_real/rebot_act_banana
robot_type: rebot
fps: 50
```

不要直接用 `data_act_real` 根目录作为LeRobot数据集本体，避免以后无法并列保存
不同任务、不同相机配置或不同数据版本。

### 4.1 采集原则

- 从训练分布内的固定初始姿态开始；
- 每条episode包含完整的接近、抓取、搬运、放置和撤离；
- 失败轨迹默认丢弃，不混入成功数据；
- 物体位置、姿态和光照做受控变化；
- 示教动作应连续，避免操作者自身产生高频抖动；
- 数据集fps必须与实际采样节拍一致；
- 不补采错过的历史帧；
- 保存前检查帧数、NaN/Inf、图像形状、相机时间差和动作范围；
- episode编号连续追加，覆盖旧数据必须显式确认。

### 4.2 初期数据量

建议分三阶段：

| 阶段 | episode数量 | 用途 |
|---|---:|---|
| 流程验证 | 3～5 | 验证采集、训练、加载和shadow链路 |
| 初始基线 | 30～50 | 判断是否能稳定学习固定场景 |
| 泛化实验 | 100以上 | 扩展物体位置、姿态和环境变化 |

少量episode只能验证软件链路，不能据此判断ACT真机泛化能力。

## 5. ACT训练设计

### 5.1 初始配置建议

已建立 `rebot_act_real/configs/act_real_banana.yaml`，当前视觉ACT基线为：

```yaml
dataset:
  repo_id: rebot_real/rebot_act_banana
  root: ./data_act_real/rebot_act_banana

policy:
  type: act
  chunk_size: 50
  n_action_steps: 25
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  use_vae: true
  latent_dim: 32
  kl_weight: 10.0
  device: cuda

output_dir: ./rebot_act_real/ckpt/act_rebot_real_banana_chunk50
batch_size: 8
steps: 50000
num_workers: 4
seed: 42
```

该配置是开发起点，不是最终最优参数。50Hz下：

```text
chunk_size=25  覆盖0.5秒
chunk_size=50  覆盖1.0秒
chunk_size=100 覆盖2.0秒
```

香蕉抓取包含连续接近、闭合和搬运阶段，首轮先用 `chunk_size=50`。后续至少对比
`25/50/100`，同时观察动作误差、夹爪时机和真机成功率。

### 5.2 `chunk_size` 与 `n_action_steps`

- `chunk_size`：模型一次预测和训练监督覆盖的动作长度；
- `n_action_steps`：不使用时间集合时，每次推理后从动作队列连续执行多少步；
- `n_action_steps` 不能大于 `chunk_size`；
- 训练数据的未来动作窗口由 `chunk_size` 决定；
- `n_action_steps` 主要决定部署时的重新推理频率。

当前真机基线使用 `n_action_steps=25`，即每500ms重新推理一次，但模型仍预测50步。
实测它比10步更能保留抓取后的连续搬运和放置意图，同时仍保留闭环重规划能力。

### 5.3 ACT时间集合的重要差异

当前LeRobot ACT启用时间集合时：

```text
temporal_ensemble_coeff 不为 None
n_action_steps 必须为 1
每个50Hz控制步都必须执行一次模型推理
```

ACT内置权重为：

```text
w_i = exp(-temporal_ensemble_coeff × i)
```

当前实现中 `i=0` 对应较旧预测，因此：

- `coeff=0`：所有预测等权；
- 正值：更偏向较旧预测；
- 负值：更偏向较新预测；
- ACT原始常用值为 `0.01`。

这与 `rebot_smolvla_real/workflow/deploy_new.py` 的自定义时间集合权重方向不同，不能
直接照搬SmolVLA的 `k=0.05` 解释。首版ACT建议先测试：

```text
temporal_ensemble_coeff=0.01
n_action_steps=1
```

只有在单次ACT推理稳定快于20ms时，才能严格维持50Hz逐步时间集合。若推理超过
20ms，部署程序不能阻塞真机控制线程；应先使用动作队列基线，或后续实现ACT专用的
异步chunk时间对齐，而不能假装模型仍在50Hz推理。

### 5.4 训练预检和输出

训练入口应在启动GPU训练前检查：

- 数据集存在且至少有一个完整episode；
- 数据集fps为50；
- 策略字段名称、形状和dtype正确；
- 双相机可正常解码；
- 关节和夹爪数据均为有限值；
- `chunk_size` 不超过有效episode长度；
- checkpoint输出目录与配置一致。

训练输出至少包含：

```text
checkpoint配置和权重
loss_curve.png
training_metrics.json
离线动作预测/真值对比图
每个动作维度的MAE
训练使用的数据集摘要和配置快照
```

离线MAE只能判断拟合情况，不能替代闭环真机成功率。

## 6. 部署推理设计

### 6.1 部署模式

部署程序计划支持：

1. `--shadow`：读取真实相机和关节，执行ACT推理但不下发动作；
2. `--execute`：经过确认后控制真机；
3. 动作队列模式：每次推理执行 `n_action_steps` 步；
4. ACT内置时间集合模式：每步推理，`n_action_steps=1`；
5. 后续可选异步ACT模式。

首次checkpoint必须先经过：

```text
数据回放 → 离线评估 → shadow → 低风险短时execute → 完整任务
```

### 6.2 真机执行顺序

每个控制周期按以下顺序处理：

```text
相机/关节观测
    ↓
ACT归一化与动作预测
    ↓
checkpoint统计反归一化
    ↓
数据集动作范围裁剪
    ↓
可选关节命令加速度整形
    ↓
SafetyGuard单步与跟踪误差保护
    ↓
机械臂位置速度命令和夹爪命令
```

ACT输出必须直接对应真机数据集的绝对动作空间。部署层不得再次进行仿真到真机的
符号转换，否则会造成二次映射。

### 6.3 加速度整形

SmolVLA真机实验已经验证以下执行层参数能消除关节命令抖动：

```text
--accel-limit
--max-accel 2.5 2.5 2.5 3.5 3.5 3.5
```

ACT首版应保留相同的可选执行层，但先分别测试：

- 原始ACT输出加 `SafetyGuard`；
- ACT时间集合；
- ACT时间集合加 `accel-limit`。

加速度整形不修改ACT任务决策，也不是阻抗控制，但过低会导致机械臂落后于ACT动作
时间轴，进而影响夹爪闭合时机。

### 6.4 运行指标

终端和可视化至少显示：

```text
控制step
单次推理耗时
实际控制频率
超时周期数
动作队列剩余步数或时间集合状态
关节反馈
ACT原始目标
安全整形后的命令
夹爪目标与反馈
相机帧龄
```

这样才能区分模型预测抖动、chunk边界跳变、执行层跟踪振荡和相机延迟。

## 7. 计划目录结构

```text
rebot_act_real/
├── README.md
├── __init__.py
├── schema.py
├── contracts.py
├── dataset_writer.py
├── configs/
│   └── act_real_banana.yaml
├── ckpt/
├── policy_logging/
│   ├── recorder.py
│   ├── analyze.py
│   ├── README.md
│   └── runs/
├── tests/
│   └── test_data_contract.py
└── workflow/
    ├── __init__.py
    ├── collect.py
    ├── inspect_dataset.py
    ├── replay.py
    ├── train.py
    ├── deploy.py
    ├── home_rebot.py
    ├── home_servo.py
    └── config/
        ├── arm.yaml
        └── gripper.yaml
```

`STservo_sdk/` 和已有机械臂配置暂时保留。开发过程中优先复用经过验证的代码，
避免同时重写硬件通信和ACT策略链路。

## 8. 运行命令

数据采集、动作重播、数据检查、ACT训练和策略部署入口均已实现。真机策略部署必须
先完成训练，并用生成的checkpoint执行shadow验证。

所有命令均在项目根目录运行：

```bash
cd /media/hjx/新加卷/hjx_ws/lerobot-mujoco-rebot
```

### 8.1 已实现：帮助和数据契约测试

查看采集参数（不会连接或控制真机）：

```bash
python -m rebot_act_real.workflow.collect --help
```

运行多模态数据契约测试：

```bash
python -m pytest -q rebot_act_real/tests/test_data_contract.py
```

### 8.2 已实现：单条短轨迹采集

首次连接硬件时，先采集100帧（50Hz下约2秒）：

```bash
python -m rebot_act_real.workflow.collect \
  --repo-id rebot_real/rebot_act_banana \
  --root ./data_act_real/rebot_act_banana \
  --task rebot_real_banana \
  --xml asset_rebot/reBot-DevArm_gripper.xml \
  --cfg rebot_act_real/workflow/config/arm.yaml \
  --gripper-cfg rebot_act_real/workflow/config/gripper.yaml \
  --port /dev/ttyUSB0 \
  --imu-port /dev/ttyUSB1 \
  --tactile-port /dev/ttyUSB2 \
  --rate 50 \
  --dataset-fps 50 \
  --episode_len 100 \
  --episodes 1
```

### 8.3 已实现：正式追加采集

```bash
python -m rebot_act_real.workflow.collect \
  --repo-id rebot_real/rebot_act_banana \
  --root ./data_act_real/rebot_act_banana \
  --task rebot_real_banana \
  --xml asset_rebot/reBot-DevArm_gripper.xml \
  --cfg rebot_act_real/workflow/config/arm.yaml \
  --gripper-cfg rebot_act_real/workflow/config/gripper.yaml \
  --port /dev/ttyUSB0 \
  --imu-port /dev/ttyUSB1 \
  --tactile-port /dev/ttyUSB2 \
  --rate 50 \
  --dataset-fps 50 \
  --episode_len 800 \
  --episodes 20
```

已有数据集存在时，程序会自动从下一条连续episode编号追加，通常不要手动传
`--episode_idx`。

需要删除已有ACT数据集并从episode 0重新采集时，在同一条正式采集命令末尾添加：

```bash
--overwrite-dataset
```

`--overwrite-dataset` 会永久删除 `--root` 指向的已有LeRobot数据集。代码会拒绝
删除缺少 `meta/info.json` 的普通目录，但执行前仍需确认路径正确。

操作方式：

```text
Enter  开始录制
s      当前episode采满且后台处理完成后保存
d      丢弃当前episode
q      安全退出
```

采集代码固定写入双相机、关节速度、夹爪反馈、IMU、磁力计、欧拉角、气压计、触觉、
各传感器帧号和时间戳。`--no-imu` 或 `--no-tactile` 只用于硬件诊断；使用后仍保留
对应字段，但内容为零值且帧号为 `-1`，不应与正式全传感器数据混合作为同一实验。

### 8.4 已实现：动作重播

仅可视化episode 0以及全部多模态数据，不连接或控制机械臂：

```bash
python -m rebot_act_real.workflow.replay \
  --root ./data_act_real/rebot_act_banana \
  --episode-idx 0 \
  --visualize-only
```

真机重播采集时实际下发的七维 `action`（终端需要输入 `REPLAY`）：

```bash
python -m rebot_act_real.workflow.replay \
  --root ./data_act_real/rebot_act_banana \
  --episode-idx 0 \
  --cfg rebot_act_real/workflow/config/arm.yaml \
  --gripper-cfg rebot_act_real/workflow/config/gripper.yaml \
  --speed-scale 1 \
  --execute
```

真机默认使用 `action[0:6]` 作为机械臂目标，并使用 `action[6]` 重播夹爪目标。
`--trajectory-key observation.state` 仅用于诊断，不是默认动作回放方式。启动真机回放
前会把机械臂在 `--prepare-duration` 指定时间内平滑移动到第一帧目标。窗口中按
`q`/`Esc` 或在终端按 `Ctrl+C` 可中止。回放结束后默认保持电机使能和最后目标；
不要添加 `--final-disable`，除非已经确认机械臂失能后不会下坠。

### 8.5 已实现：数据检查

```bash
python -m rebot_act_real.workflow.inspect_dataset \
  --root ./data_act_real/rebot_act_banana \
  --image-check all
```

完整检查并保存JSON报告：

```bash
python -m rebot_act_real.workflow.inspect_dataset \
  --root ./data_act_real/rebot_act_banana \
  --image-check all \
  --json-report ./data_act_real/rebot_act_banana_report.json
```

快速抽样解码可以将 `--image-check all` 改成 `sample`；`none` 只检查图像字节存在，
不实际解码。添加 `--strict` 后，warning也会返回非零退出码。

检查器会流式检查episode连续性、帧索引、双相机图像、关节状态、动作、夹爪反馈、
IMU、触觉、帧号和传感器时间戳，不会一次把整个数据集解码到内存。50Hz数据采样
读取低于50Hz的相机最新帧时，连续图像或相机frame ID重复属于预期现象；应结合
`unique_frames`、相机实际帧率和是否出现长时间完全卡帧判断，不能仅凭重复帧warning
认定数据损坏。

### 8.6 已实现：训练

```bash
python -m rebot_act_real.workflow.train --check-only
python -m rebot_act_real.workflow.train
```

默认配置使用双相机、六关节状态和七维动作，`chunk_size=50`、
`n_action_steps=25`，训练50000步。训练结果保存到：

```text
rebot_act_real/ckpt/act_rebot_real_banana_chunk50/
```

### 8.7 已实现：动作队列shadow

```bash
python -m rebot_act_real.workflow.deploy \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --shadow \
  --rate 50 \
  --n-action-steps 25 \
  --visualize
```

动作队列模式每次网络推理产生50步，但只执行前25步，然后用最新观测重新推理。
终端 `queue` 表示当前队列剩余动作数，`inference` 只在队列重新填充时明显升高。

### 8.8 已实现：动作队列真机部署

首次真机策略部署建议先使用动作队列基线：

```bash
python -m rebot_act_real.workflow.deploy \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --execute \
  --rate 50 \
  --n-action-steps 20 \
  --accel-limit \
  --max-accel 2.5 2.5 2.5 3.5 3.5 3.5 \
  --visualize
```

终端需要输入 `DEPLOY`。`accel-limit` 只整形最终关节命令，不改变ACT策略目标。
当前香蕉抓取放置真机实验中，`n_action_steps=20` 已成功完成抓取、搬运和放置。
20步在50Hz下连续执行0.4秒，比原先10步的0.2秒更能保留ACT动作块中的连续搬运
意图，同时仍然每0.4秒根据最新双相机和关节观测重新规划。

### 8.9 已实现：ACT原生时间集合实验（当前真机实测推荐）

先在shadow模式测量每步推理耗时：

```bash
python -m rebot_act_real.workflow.deploy \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --shadow \
  --rate 50 \
  --temporal-ensemble \
  --temporal-ensemble-coeff 0.01 \
  --visualize
```

LeRobot ACT原生时间集合会强制 `n_action_steps=1` 并在每个控制周期执行一次完整模型
推理。只有 `inference` 能稳定低于20ms且 `hz` 接近50、`overrun` 不持续增长时，
才适合进入真机实验。否则应使用动作队列模式，不能通过降低控制频率伪装成50Hz。

### 8.10 已实现：ACT实时MIT关节阻抗部署

专用入口 `deploy_impedance_control.py` 保持ACT作为唯一目标动作来源，将ACT的六关节
绝对目标通过实时MIT位置、速度、刚度、阻尼和附加力矩接口下发。阻抗层不生成轨迹，
也不会替代ACT视觉闭环。该入口固定使用ACT原生时间集合：
`n_action_steps=1`，每个控制步都执行一次完整策略推理，不使用多步动作队列。
`--n-action-steps` 和 `--temporal-ensemble` 仅为兼容旧命令保留，不能关闭上述行为。

先运行shadow确认策略加载和相机稳定；shadow不会使能机械臂，因此不实际测试阻抗：

```bash
python -m rebot_act_real.workflow.deploy_impedance_control \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --shadow \
  --rate 50 \
  --temporal-ensemble-coeff 0.01 \
  --visualize
```

以下参数已在香蕉抓取放置真机实验中验证运行流畅，作为当前推荐配置：

```bash
python -m rebot_act_real.workflow.deploy_impedance_control \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --execute \
  --rate 50 \
  --temporal-ensemble-coeff 0.01 \
  --mit-kp 16 16 16 12 12 12 \
  --mit-kd 0.1 0.3 0.3 0.2 0.1 0.1 \
  --mit-kp-ramp-sec 0 \
  --accel-limit \
  --max-accel 2 2 2 3 3 3 \
  --visualize \
  --yes
```

该脚本默认启用 `--impedance-control`，默认关闭目标速度前馈，避免ACT chunk边界的
位置变化被放大为速度冲击。需要对照原始位置速度执行层时可传
`--no-impedance-control`。

阻抗部署默认启用与SmolVLA参考代码相同的Pinocchio实时重力前馈，无需额外添加
`--gravity-compensation`。当前 `lerobot_rebot` 环境的动力学扩展如果仍链接到
Python 3.8版Boost.Python，则可能与当前Python 3.10不兼容；脚本会在连接相机和
机械臂之前完成预检，不可用时安全退出。`--no-gravity-compensation` 只建议用于
定位动力学环境问题，不建议用于真机任务。重力补偿参数可按需覆盖：

```bash
--gravity-compensation \
--gravity-scales 1.5 1 0.95 0.85 1 1 \
--torque-limits 10 10 10 5 5 5
```

低刚度关节阻抗可以降低接触时的位置刚性和冲击，但不是碰撞检测器，不能保证避免
碰撞，也不能替代动作范围、单步限幅、跟踪误差保护和物理急停。

### 8.11 可选IMU/触觉多模态ACT与消融实验

数据集中的双相机和六关节状态训练逻辑保持不变。传入 `--imu` 后增加10维
`sensor.imu` 分支（标准化、MLP编码）；传入 `--tactile` 后增加
`sensor.tactile` 分支（逐单元标准化、保持12x30布局的二维CNN编码）。各分支输出
64维表示，拼接为独立传感器token送入ACT Transformer。不要把原始传感器直接拼到
六维 `observation.state`，否则会改变机器人状态语义并破坏纯视觉基线的可比性。

四组消融训练命令：

```bash
# 相机+关节基线，保持原有输出目录
python -m rebot_act_real.workflow.train

# 相机+关节+IMU，自动输出到 ...chunk50_imu
python -m rebot_act_real.workflow.train --imu

# 相机+关节+触觉，自动输出到 ...chunk50_tactile
python -m rebot_act_real.workflow.train --tactile

# 相机+关节+IMU+触觉，自动输出到 ...chunk50_imu_tactile
python -m rebot_act_real.workflow.train --imu --tactile
```

仅做数据与形状预检时添加 `--check-only`。编码维度可通过
`--sensor-embed-dim 64` 调整；消融实验应固定该参数、随机种子、训练步数、数据集、
chunk size和部署参数，只改变模态组合。

多模态checkpoint会额外保存 `rebot_multimodal.json`，记录启用的模态和编码维度。
部署时参数必须和checkpoint完全一致，否则程序拒绝启动，防止串用权重：

```bash
# IMU checkpoint
python -m rebot_act_real.workflow.deploy_impedance_control \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50_imu/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --imu --imu-port /dev/ttyUSB1 \
  --execute --rate 50 --temporal-ensemble-coeff 0.01 \
  --mit-kp 16 16 16 12 12 12 --mit-kd 0.1 0.3 0.3 0.2 0.1 0.1 \
  --mit-kp-ramp-sec 0 --accel-limit --max-accel 2 2 2 3 3 3 \
  --visualize --yes

# IMU+触觉 checkpoint
python -m rebot_act_real.workflow.deploy_impedance_control \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50_imu_tactile/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --imu --tactile --imu-port /dev/ttyUSB1 --tactile-port /dev/ttyUSB2 \
  --execute --rate 50 --temporal-ensemble-coeff 0.01 \
  --mit-kp 16 16 16 12 12 12 --mit-kd 0.1 0.3 0.3 0.2 0.1 0.1 \
  --mit-kp-ramp-sec 0 --accel-limit --max-accel 2 2 2 3 3 3 \
  --visualize --yes
```

部署会检查IMU/触觉帧号、时间新鲜度、形状和NaN/Inf。多模态传感器异常时停止本次
动作推理，不会使用全零数据静默替代。普通 `deploy.py` 同样支持 `--imu` 和
`--tactile`，但阻抗真机实验建议使用上面的专用入口。启用 `--visualize` 时，窗口
右侧同步显示IMU四元数、角速度、加速度和模长，以及12x30触觉热力图、最大值、均值
和接触压力中心；左侧原有双相机、关节状态和动作曲线保持不变。
程序会在连接和使能机械臂之前等待FlexiTac完成无接触baseline，并确认IMU与触觉
均已输出有效首帧。默认等待15秒，设备启动较慢时可传
`--sensor-ready-timeout-s 30`；baseline期间不要触碰触觉阵列。

### 8.12 策略部署实验记录与论文分析

普通ACT入口 `workflow.deploy` 和MIT阻抗入口
`workflow.deploy_impedance_control` 使用同一套实验记录格式。记录功能默认关闭，
不会在普通部署时创建目录或产生额外磁盘I/O。正式实验必须显式添加：

```bash
--record --max-steps 800 --run-name task_method_seed01
```

其中：

- `--record`：开启状态、动作、时序和多模态传感器记录；
- `--max-steps`：控制循环最大步数，达到后正常退出并刷新全部文件；
- `--run-name`：论文实验名称，建议包含任务、方法/模态和随机种子；
- `--record-images`：可选保存双相机视频，必须与 `--record` 同时使用；
- `--record-root`：可选修改实验记录根目录；
- `--record-chunk-size`：数据和视频分段长度，默认250个控制步；
- `--record-queue-size`：异步写盘队列容量，默认1000。

`--max-steps` 统计控制循环而不是墙钟秒数。在50Hz名义频率下：

| 参数 | 名义运行时长 |
|---:|---:|
| `--max-steps 800` | 约16秒 |
| `--max-steps 1500` | 约30秒 |
| `--max-steps 3000` | 约60秒 |
| `--max-steps 0` | 不限制 |

如果发生推理超时或控制周期overrun，实际墙钟时长可能更长。论文中应使用记录的
`duration_s` 和 `loop_dt_s` 报告真实运行时间和控制频率。固定步数实验达到上限后
会正常执行清理流程，无需手动按 `Ctrl+C`。

普通ACT、IMU+触觉、时间集合、固定800步并保存视频的完整命令：

```bash
python -m rebot_act_real.workflow.deploy \
  --checkpoint rebot_act_real/ckpt/act_rebot_real_banana_chunk50_imu_tactile/checkpoints/last/pretrained_model \
  --dataset-root ./data_act_real/rebot_act_banana \
  --repo-id rebot_real/rebot_act_banana \
  --execute \
  --rate 50 \
  --temporal-ensemble \
  --temporal-ensemble-coeff 0.01 \
  --imu \
  --tactile \
  --imu-port /dev/ttyUSB1 \
  --tactile-port /dev/ttyUSB2 \
  --sensor-ready-timeout-s 30 \
  --visualize \
  --yes \
  --record \
  --record-images \
  --max-steps 800 \
  --run-name banana_video_seed01
```

不需要定性视频时删除 `--record-images`，这样可显著降低写盘带宽，更适合正式的
50Hz时延和控制性能实验。

#### 8.12.1 保存位置与格式

默认实验根目录为：

```text
rebot_act_real/policy_logging/runs/
```

每次运行会创建带时间戳的唯一目录：

```text
rebot_act_real/policy_logging/runs/
└── 20260728_181503_banana_video_seed01/
    ├── metadata.json
    ├── summary.json
    ├── data/
    │   ├── chunk_000000.npz
    │   ├── chunk_000001.npz
    │   └── ...
    ├── videos/
    │   ├── cam_high_000000.mp4
    │   ├── cam_wrist_000000.mp4
    │   └── ...
    └── figures/
        ├── trajectory.png
        ├── timing.png
        ├── multimodal.png
        ├── keyframes.png
        ├── overview.png
        ├── tactile_analysis.png
        ├── imu_analysis.png
        ├── keyframes.json
        └── metrics.json
```

原始记录包含：

- 六关节反馈位置和估算速度；
- ACT原始七维动作与安全整形后实际命令；
- 逐关节跟踪误差；
- 推理耗时、推理序号、动作年龄和推理频率；
- 控制周期、实际控制频率、overrun累计数和动作队列长度；
- 双相机帧号及相对单调时钟时间戳；
- 启用时的IMU、12×30触觉阵列、帧号和采样时间戳；
- MIT阻抗模式下的重力补偿力矩、最终前馈力矩和位置误差；
- 完整命令行、checkpoint、数据集路径、Git commit、主机环境、单位和模态配置。

NPZ采用异步、压缩、分块和原子提交方式，避免磁盘操作直接阻塞控制线程。视频同样
按数据chunk保存为可独立播放的短MP4，并在完成编码和关闭后才原子提交。因此强制
中断最多影响尚未提交的最后一段，之前的视频分段仍可播放。隐藏的
`.partial.mp4` 表示未完成分段，不应作为有效实验视频。

正常情况下应等待固定步数自动结束，或者只按一次 `Ctrl+C` 并等待终端显示：

```text
实验记录已保存: ...
```

连续两次 `Ctrl+C` 会触发强制退出，只应用于程序无法正常结束的情况。

#### 8.12.2 生成论文图和定量指标

对一次完成的运行执行：

```bash
python -m rebot_act_real.policy_logging.analyze \
  --run rebot_act_real/policy_logging/runs/20260728_181503_banana_video_seed01
```

默认结果保存在该实验的 `figures/` 目录。也可以指定独立输出目录：

```bash
python -m rebot_act_real.policy_logging.analyze \
  --run rebot_act_real/policy_logging/runs/20260728_181503_banana_video_seed01 \
  --output ./paper_results/banana_video_seed01
```

分析程序输出适合论文排版的300dpi PNG以及 `metrics.json`，不生成PDF。
指标包括推理时延的均值/P50/P95/P99/最大值、控制周期均值与抖动、overrun帧数、
逐关节跟踪RMSE和最大绝对误差，以及启用触觉时的响应峰值和均值。绘图使用
色盲友好配色、统一线宽和无上/右边框风格。

存在双相机视频时，分析器还会生成 `keyframes.png` 定性拼图。两行分别对应
场景相机和腕部相机，各列严格对齐到同一个控制步。关键时刻根据任务开始/结束、
夹爪动作变化、机械臂动作变化、触觉峰值和最大跟踪误差自动选择；相近事件会去重，
并用任务进程中的代表帧补足。`keyframes.json` 保存每列画面的控制步、相对时间、
选帧依据和对应指标值，便于撰写论文图注以及核对自动选帧是否符合任务语义。

`overview.png` 是面向论文主文的综合总览大图，在统一网格中整合双相机关键帧、
关节反馈与命令、闭环跟踪误差、夹爪动作、推理/控制实时性，并按实际记录内容动态
加入IMU动态、触觉响应和触觉峰值热力图。未启用或未记录的模态不会显示，也不会
留下空的占位面板。

存在触觉数据时还会单独生成 `tactile_analysis.png`：联合展示接触强度与检测阈值、
活跃taxel面积、压力中心随时间变化、多个控制步的12×30触觉快照、峰值三维触觉
表面、时空接触演化、压力中心轨迹，以及有视频时与触觉峰值同步的场景/腕部画面。
`metrics.json` 会增加接触持续时间、峰值时刻、最大接触面积、峰值中心位置和中心
轨迹长度等触觉定量指标。没有触觉模态时不会生成该图。

存在IMU数据时还会单独生成 `imu_analysis.png`：通过 `wxyz` 四元数恢复传感器姿态，
展示三维单位球姿态轨迹、四元数和相对旋转角、三轴角速度/线加速度、姿态坐标架
序列、角速度频谱、角加速度与jerk运动强度，以及有视频时与角速度峰值同步的双相机
画面。采用四元数而不是连续欧拉角作为主姿态表示，避免接近欧拉角奇异点时产生
误导性跳变。没有IMU模态时不会生成该图。

更详细的字段和数据约定见 `rebot_act_real/policy_logging/README.md`。

## 9. 实施阶段

### 阶段A：数据层

- 建立ACT独立schema和dataset writer；
- 移植并精简50Hz真机采集；
- 完成数据检查、回放和契约测试；
- 采集3～5条流程验证数据。

验收条件：数据集可重新加载，双相机可解码，字段和频率正确，episode可回放。

### 阶段B：训练层

- 建立ACT YAML配置；
- 实现训练预检；
- 训练小规模checkpoint；
- 输出loss、动作误差和配置快照。

验收条件：checkpoint能通过 `ACTPolicy.from_pretrained` 独立加载，离线推理维度和
动作单位正确。

### 阶段C：部署层

- 实现shadow；
- 实现动作队列基线；
- 实现ACT内置时间集合；
- 接入SafetyGuard、动作范围裁剪和可选加速度整形；
- 增加真机可视化与性能指标。

验收条件：shadow稳定运行，控制循环不因推理或相机阻塞，真机命令始终处于数据集
范围和安全限幅内。

### 阶段D：策略研究

- 对比 `chunk_size=25/50/100`；
- 对比动作队列与时间集合；
- 对比 `temporal_ensemble_coeff`；
- 对比是否使用腕部相机；
- 对比是否启用加速度整形；
- 使用固定测试场景统计成功率、完成时间和夹爪时机。

### 阶段E：多模态扩展

视觉ACT基线稳定后，再决定是否将IMU或触觉加入 `observation.*` 策略输入。扩展时
必须建立新数据集版本和新checkpoint，不能在已有视觉数据集上静默改变字段语义。

## 10. 安全要求

- 首次运行任何checkpoint必须使用shadow；
- 真机执行前确认物理急停可用；
- 工作空间内不得有人；
- 从训练分布内初始姿态启动；
- checkpoint、数据统计和相机字段必须匹配；
- 发现NaN、动作越界、跟踪误差持续超限或相机超时应立即停止；
- 默认保留单步限幅和跟踪误差保护；
- 不在首版ACT中加入阻抗控制；
- 每次只改变一个策略或执行参数，并记录对应实验结果。
