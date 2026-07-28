# reBot SmolVLA 真机工作区

## 1. 目标

本目录用于在当前 `lerobot-mujoco-rebot` 项目中建立独立、可逐步验证的
SmolVLA 真机工作流。第一阶段只实现最小闭环：

```text
双相机图像 + 六轴关节状态 + 自然语言
                    ↓
                SmolVLA
                    ↓
六轴绝对关节目标 + 夹爪目标
```

首个任务示例：

```text
抓取香蕉并放置到盘子中。
```

采集阶段同时保留 IMU 和触觉，避免以后重新采集示教；第一阶段 SmolVLA
训练仍只使用双相机、语言、六轴状态和七维动作。

## 2. 已有代码的角色

当前项目已经具备真机工作流需要的大部分底层能力：

| 能力 | 现有来源 | 新工作流中的用途 |
|---|---|---|
| Astra-S 第三视角 | `rebot_scripts/Servo_control/astra_s_shm_server.py` | `observation.image` |
| D405 腕部视角 | `ThreadedRealSenseCamera` | `observation.wrist_image` |
| 舵机主手读取 | `record_rebot_episodes.py` | 示教目标生成 |
| reBot 电机控制 | `reBotArm_control_py` + `motorbridge` | 真机动作执行 |
| 夹爪 MIT 控制 | `setup_damiao_gripper` / `send_damiao_gripper_mit` | 第 7 维动作 |
| 安全限位与跟踪 | `SafetyGuard` | 采集和部署保护 |
| SmolVLA 训练 | `train_model.py`、`scripts/08_smolvla.py` | 直接复用 |

`act_tactile_rebot` 的 HDF5、ACT、IMU 和触觉网络不会直接搬入 SmolVLA。
需要复用的是其经过真机验证的硬件访问、安全控制、相机并发和清理流程。

## 3. 第一阶段数据契约

真机数据直接写入 LeRobot 数据集，不再先保存 ACT 风格 HDF5 后转换。

每一帧包含：

| Feature | 形状 | 类型 | 含义 |
|---|---:|---|---|
| `observation.image` | `(256,256,3)` | `uint8` | Astra-S RGB |
| `observation.wrist_image` | `(256,256,3)` | `uint8` | D405 RGB |
| `observation.state` | `(6,)` | `float32` | 真机六轴反馈角 |
| `action` | `(7,)` | `float32` | 六轴绝对目标角 + 夹爪目标 |
| `sensor.joint_velocity` | `(6,)` | `float32` | 六轴反馈速度 |
| `sensor.gripper_feedback` | `(2,)` | `float32` | 夹爪位置和速度反馈 |
| `sensor.imu` | `(10,)` | `float32` | 四元数、角速度和加速度 |
| `sensor.imu_magnetometer` | `(3,)` | `float32` | 三轴磁力计 |
| `sensor.imu_euler` | `(3,)` | `float32` | IMU 欧拉角 |
| `sensor.imu_barometer` | `(4,)` | `float32` | IMU 气压计原始输出 |
| `sensor.tactile` | `(12,30)` | `float32` | 右夹爪 FlexiTac |
| `sensor.frame_ids` | `(4,)` | `int64` | 两相机、IMU、触觉帧号 |
| `sensor.timestamps` | `(4,)` | `float64` | 四路传感器单调时钟 |
| `task` | 文本 | string | 当前自然语言任务 |

标准定义位于 [schema.py](schema.py)。

### 3.1 状态定义

第一阶段保持与仿真语言数据一致：

```text
observation.state = [q1, q2, q3, q4, q5, q6]
```

夹爪反馈、IMU 和触觉使用 `sensor.*` 辅助字段保存。LeRobot 的
`dataset_to_policy_features()` 会忽略这个命名空间，因此不会改变当前 SmolVLA
输入或 checkpoint 结构。

### 3.2 动作定义

```text
action = [q1_target, ..., q6_target, gripper_target]
```

- 前 6 维使用真机控制器最终下发的绝对关节目标，单位为弧度；
- 第 7 维使用真机达妙夹爪最终下发的目标弧度；
- 数据采集和部署必须使用完全相同的单位、方向及开闭定义；
- 不把主手舵机角、归一化夹爪量或MuJoCo夹爪位移混入训练动作。

这与仿真在语义上保持一致：策略输出“六轴目标 + 夹爪目标”，但真机夹爪数值
使用真实电机空间。模型通过数据集统计进行归一化，部署时反归一化后可直接得到
真机目标，因此不应复用仿真中的 `0.001/0.05 → 0/1` 适配器。

### 3.3 时间对齐

在数据采样时刻 `t`：

```text
observation(t) = 控制命令下发前的最新相机帧和关节反馈
action(t)      = 本次控制循环最终下发的关节与夹爪目标
```

必须保存“真正通过安全限制、平滑和限幅后下发的动作”，不能保存未经处理的主手
原始命令，否则训练动作和机器人真实运动不一致。

## 4. 目标工作流

### 4.1 硬件预检

依次验证：

1. Astra-S 能稳定输出RGB；
2. D405 能稳定输出RGB；
3. 两相机时间戳和帧号持续更新；
4. 六轴反馈读取正常；
5. 夹爪反馈与命令方向正确；
6. 急停、异常失能和回零流程正常。

相机或反馈无效时禁止开始记录episode。

### 4.2 语言示教采集

`workflow/collect.py` 的职责为：

1. 读取命令行中的数据集、任务和硬件参数；
2. 初始化主手、真机、夹爪和双相机；
3. 等待操作者确认；
4. 以50Hz执行真机遥操作；
5. 以50Hz抽取同步训练帧；
6. 调用`LeRobotDataset.add_frame(..., task=instruction)`；
7. 成功时保存episode，失败或异常时清空episode buffer；
8. 安全关闭相机、串口和电机。

每个 episode 只保存一条人工确认成功的完整轨迹。录制结束后必须输入 `s`
保存；输入 `d` 会删除临时图像并清空 episode buffer。

启动示例：

```bash
python -m rebot_smolvla_real.workflow.collect \
  --repo-id rebot_real/rebot_smolvla_multimodal \
  --root ./data_vla_real/rebot_smolvla_multimodal \
  --instruction "抓取香蕉并放置到盘子中" \
  --xml asset_rebot/reBot-DevArm_gripper.xml \
  --cfg rebot_smolvla_real/workflow/config/arm.yaml \
  --gripper-cfg rebot_smolvla_real/workflow/config/gripper.yaml \
  --port /dev/ttyUSB0 --imu-port /dev/ttyUSB1 --tactile-port /dev/ttyUSB2 \
  --rate 50 --dataset-fps 30 --episode_len 600 \
  --episodes 20 --episode_idx 0
```

运行时 `Enter` 开始，`s` 保存，`d` 丢弃，`q` 安全退出。第一次
`Ctrl+C` 会执行安全清理；如果底层硬件驱动阻塞，第二次 `Ctrl+C` 会强制退出。
任一相机没有有效新帧时会拒绝录制，录制中相机失步会自动丢弃当前 episode。
临时不接 IMU 或触觉时可添加 `--no-imu` 或 `--no-tactile`，对应字段会保存
固定形状零值。

开始录制后终端会显示当前 `episode_idx` 的实时帧进度、采集速度、已用时间和
预计剩余时间。进度以 `--dataset-fps` 保存帧计数，而不是以 50Hz 控制循环计数。

采集数量和编号参数含义如下：

- `--episodes 20` 表示本次运行成功保存20条轨迹后自动结束；默认值 `0`
  表示不限制数量。按 `d` 丢弃的轨迹不计数，也不占用编号。
- `--episode_idx 0` 表示本次第一条轨迹从编号0开始。每按 `s` 成功保存一条，
  编号自动加1。因此空数据集配合 `--episodes 20 --episode_idx 0` 会生成
  episode `0` 到 `19`。
- 向已有数据集追加时，`--episode_idx` 必须等于数据集下一连续编号。例如已经
  保存 `0` 到 `9`，下一次应传 `--episode_idx 10`。也可以省略
  `--episode_idx`，由程序自动读取下一编号，这是追加采集时更安全的用法。
- `--episodes` 与 `--episode_idx` 不冲突：前者控制本次成功保存的数量，后者
  控制本次第一条数据的编号。编号与现有数据不连续时，程序会在硬件初始化前报错。

默认不会覆盖已有数据。如需明确删除 `--root` 下的旧 LeRobot 数据集并从0重新
采集，使用：

```bash
python -m rebot_smolvla_real.workflow.collect \
  --root ./data_vla_real/rebot_smolvla_multimodal \
  --episodes 20 --episode_idx 0 \
  --overwrite-dataset \
  [其余硬件参数]
```

`--overwrite-dataset` 是破坏性参数：旧数据、视频和元数据都会被永久删除。为防止
误删普通目录，目标目录必须是包含 `meta/info.json` 的 LeRobot 数据集；覆盖时
`--episode_idx` 只能省略或设为 `0`。

### 4.2.1 数据集完整性检查

采集后可运行只读检查脚本，它不会连接或控制真机，也不会修改数据集：

```bash
python -m rebot_smolvla_real.workflow.inspect_dataset
```

默认检查 `data_vla_real/rebot_smolvla_multimodal`，也可以明确指定数据集并保存
JSON报告：

```bash
python -m rebot_smolvla_real.workflow.inspect_dataset \
  --root ./data_vla_real/rebot_smolvla_multimodal \
  --image-check all \
  --json-report ./data_vla_real/rebot_smolvla_multimodal_report.json
```

报告包含：

- 数据集磁盘占用、文件分类大小及完全解码后的估算内存；
- LeRobot版本、机器人类型、帧率、episode数、总帧数、时长和语言任务；
- Parquet文件、episode编号、帧数、全局index和frame index连续性；
- 两路相机图像缺失、损坏、分辨率错误、重复帧和独立帧数量；
- 本体状态、动作、关节速度、夹爪、IMU和12×30触觉的shape、NaN/Inf、
  全零比例、最小值、最大值、均值和标准差；
- 四路传感器frame ID、时间戳递增情况和最大同步偏差。

`--image-check all` 会解码全部图像，检查最完整；`sample` 每秒抽查一帧图像；
`none` 只检查图像字节是否存在。脚本按Parquet row group流式读取，不会一次将
整个数据集加载到内存。发现结构损坏或数据缺失时返回非零退出码；普通warning
默认仍返回0，添加 `--strict` 后warning也会返回非零退出码。

### 4.2.2 多传感器轨迹重播

先使用只可视化模式检查 episode，不会连接或控制机械臂：

```bash
python -m rebot_smolvla_real.workflow.replay \
  --root ./data_vla_real/rebot_smolvla_multimodal \
  --episode-idx 0 \
  --visualize-only
```

确认轨迹正确后才进行真机重播：

```bash
python -m rebot_smolvla_real.workflow.replay \
  --root ./data_vla_real/rebot_smolvla_multimodal \
  --episode-idx 0 \
  --cfg rebot_smolvla_real/workflow/config/arm.yaml \
  --gripper-cfg rebot_smolvla_real/workflow/config/gripper.yaml \
  --execute
```

真机模式需要在终端输入 `REPLAY` 二次确认。默认使用数据集中的 `action`
作为六轴和夹爪目标，并保留余弦预引导、单步限幅和跟踪误差保护。回放结束或
中止后默认**不调用 `arm.disable()`**，机械臂与夹爪保持使能和最后目标，避免
突然失去支撑而下坠。之后应由回零程序或其他控制程序平稳接管。只有明确需要卸载
力矩且有人扶稳机械臂时，才可显式添加 `--final-disable`。
可视化窗口同步显示：

- Astra-S 第三视角和 D405 腕部图像；
- IMU 四元数、角速度、加速度、磁力计、欧拉角和气压数据；
- 12×30 FlexiTac 热力图；
- 关节位置、速度、动作和夹爪反馈；
- IMU、触觉、夹爪历史曲线；
- 各传感器帧号和相对时间偏差。

窗口中按 `q` 或 `Esc`，或者在终端按 `Ctrl+C`，均可中止重播。可以先用
`--speed-scale 0.5` 做低速真机检查。

### 4.3 数据检查

后续 `inspect_dataset.py` 至少检查：

- 两相机图像是否错位、卡帧或颜色通道颠倒；
- 每个episode长度和控制频率；
- 6维状态和7维动作是否有限；
- 状态与动作是否存在一帧以上异常延迟；
- 夹爪开闭样本比例及转换时刻；
- 语言文本是否与任务一致；
- 轨迹是否碰撞桌面、盘子或非目标物体。

训练前必须回放全部episode。

### 4.4 训练

数据格式与现有SmolVLA训练入口兼容，后续增加独立训练配置：

```text
python -m scripts.08_smolvla train \
  --config rebot_smolvla_real/configs/smolvla_real_banana.yaml
```

训练配置已经建立，但在产生并检查真机数据集之前不应启动训练。

第一版建议分阶段保存和真机评估checkpoint，不只依据训练loss选择模型。

### 4.5 真机部署

后续 `deploy.py` 的闭环为：

```text
获取最新双相机 + 六轴反馈
        ↓
构建 SmolVLA batch（含task）
        ↓
生成动作chunk并反归一化
        ↓
关节/夹爪安全检查、限速、限幅和平滑
        ↓
下发动作并持续监控反馈
```

部署必须保留旧真机代码中的：

- 启动保持与动作渐入；
- 关节范围限制；
- 单步最大变化；
- 跟踪误差累计保护；
- 人工干预检测；
- 相机超时保护；
- 异常时停止下发并安全退出。

默认部署仍采用SmolVLA自身的动作队列。真机出现chunk边界不连续时，可显式启用
ACT风格的时间集合：保存带起始控制步的重叠动作chunk，只融合指向同一控制时刻的
动作，因此不会把未来动作提前执行或将历史动作顺延。该功能使用同步推理，不启动
后台线程；实际可用推理间隔仍应根据GPU上的单次推理耗时确定。

## 5. 实施阶段

### 阶段A：接口冻结

- [x] 建立独立工作区；
- [x] 固定双相机、6维状态、7维动作和语言数据契约；
- [x] 增加配置示例和帧校验工具；
- [x] 增加配置加载、同步样本和LeRobot写入组件；
- [x] 增加SmolVLA真机训练配置；
- [ ] 明确真机夹爪闭合/张开的实际弧度范围；
- [ ] 测量两相机、反馈和控制的实际频率。

### 阶段B：采集器

- [ ] 将相机类从超长ACT脚本中提取为可复用模块；
- [ ] 将主手和真机控制提取为硬件适配器；
- [x] 实现LeRobotDataset直接采集；
- [x] 实现成功保存、失败丢弃和episode语言元数据；
- [x] 同步保存IMU、触觉、反馈速度和传感器时序诊断字段；
- [ ] 完成至少一条香蕉任务数据的离线回放。

### 阶段C：训练

- [x] 增加SmolVLA真机训练YAML和独立训练入口；
- [ ] 检查dataset stats和夹爪分布；
- [ ] 训练并保存阶段checkpoint；
- [ ] 做离线逐维动作误差和夹爪切换评估。

### 阶段D：安全部署

- [x] 实现只读取、不下发动作的shadow模式；
- [x] 实现关节动作限幅、数据范围裁剪和跟踪保护；
- [ ] 实现夹爪命令转换和迟滞（仅在真实数据表明确有需要时）；
- [ ] 空载低速测试；
- [ ] 单物体、固定位置闭环测试；
- [ ] 统计抓取、放置、碰撞和急停次数。

### 阶段E：扩展模态

基础VLA工作流稳定后，再考虑：

- 将IMU投影为额外状态token；
- 将12×30触觉阵列编码为触觉token；
- 使用接触信息辅助夹爪闭合和滑移检测；
- 研究视觉、语言、状态与触觉的联合动作专家。

IMU和触觉扩展会改变模型输入与checkpoint结构，不应混入第一阶段。

## 6. SmolVLA 真机训练

当前训练配置：

```text
rebot_smolvla_real/configs/smolvla_real_banana.yaml
```

先检查数据集、PyArrow兼容性和策略字段：

```bash
python -m rebot_smolvla_real.workflow.train --check-only
```

开始训练：

```bash
python -m rebot_smolvla_real.workflow.train
```

默认配置为20万步、batch size 4，每1万步保存checkpoint，输出到：

```text
rebot_smolvla_real/ckpt/smolvla_rebot_real_banana
```

训练只使用两路相机、语言、六轴状态和七维动作。所有 `sensor.*` 字段仍保存在
数据集中，但不会成为SmolVLA输入。当前项目固定使用：

```bash
pip install pyarrow==19.0.1
```

这是因为 `datasets==3.4.1` 与当前环境中的 PyArrow 23 在解码12×30触觉扩展
字段时不兼容。

## 7. SmolVLA 真机部署

训练完成后先运行shadow模式。该模式读取真机状态和双相机并执行模型推理，但不会
使能机械臂或发送动作：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --shadow \
  --inference-every 1 \
  --noise-correlation 0.7 \
  --chunk-takeover-steps 1 \
  --rate 50 \
  --visualize
```

先观察终端中的 `inference=...ms`、`ensemble=...`、关节曲线和相机画面。确认预测
值与推理速度正常后，再使用同一组时间集合参数启用真机动作：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --execute \
  --inference-every 1 \
  --noise-correlation 0.7 \
  --chunk-takeover-steps 1 \
  --rate 50 \
  --visualize
```

`--execute` 还需要输入 `DEPLOY` 二次确认。策略输出直接解释为：

```text
[六轴真机绝对目标rad, 达妙夹爪目标rad]
```

部署不会使用仿真中的夹爪二值映射。默认启用数据动作范围裁剪、关节最大单步变化、
连续跟踪误差保护、夹爪最大单步变化和相机超时保护。退出后默认保持机械臂使能，
避免突然下坠；之后应运行回零程序接管。只有显式传入 `--final-disable` 才会失能。

时间集合参数说明：

- `--temporal-ensemble`：启用ACT风格的同一时刻重叠动作融合；不传时维持原始
  SmolVLA动作队列行为。
- `--temporal-ensemble-k 0.2`：指数衰减系数。值越大越信任最新预测，值越小越
  平滑。
- `--temporal-ensemble-history 5`：最多保留5个预测chunk。当前checkpoint的
  `chunk_size=5`。
- `--inference-every 1`：每个控制步生成新chunk，重叠最充分，但同步推理负载最高。
- `--inference-every 2`：建议的初始测试值；在5步chunk下最多约有3个对齐预测。
- `--inference-every 4`：计算负载更接近原始每5步推理一次，主要改善chunk边界。
- `--gripper-latest-action`：夹爪使用最新chunk的对齐动作，避免开合命令被平均后
  响应变慢。

时间集合计算本身没有增加等待、异步队列或动作时间偏移，但同步模型推理耗时依然
存在。50Hz控制周期只有20ms；如果单次推理超过20ms，`--inference-every 1` 无法
严格维持50Hz。应先在shadow模式测量，再选择 `1`、`2` 或 `4`。推理间隔不能超过
checkpoint的 `chunk_size`，代码会在启动时检查这一条件。

### 7.1 异步低抖动部署

`workflow/deploy_new.py` 默认启用：

- 最新观测双缓冲异步推理：未处理的旧观测会被覆盖，不形成推理积压；
- 相邻Flow-Matching随机噪声默认使用 `rho=0.7` 的相关序列，在保持单位高斯
  方差的同时减少跨chunk随机跳变；固定噪声仅作为显式实验选项；
- 控制循环不等待GPU，新chunk完成后从第一个动作开始执行；
- 保持原始5步、每步20ms的动作时间尺度，只对新chunk前两个关节目标做
  40%和80%新目标权重的短接管；第3步起完全使用模型原动作，夹爪不参与短接管；
- 原始关节执行层：保留 `vlim`、`max-step` 和 `SafetyGuard`，默认不增加
  加速度整形；
- 夹爪时间集合、0.01rad死区、0.7低通和0.15rad单步限制；
- 实际循环频率 `hz`、超时累计值 `overrun` 和推理观测年龄 `age`。

先运行shadow：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --shadow \
  --inference-every 1 \
  --rate 50 \
  --chunk-takeover-steps 1 \
  --visualize
```

确认后运行真机：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --execute \
  --inference-every 1 \
  --rate 50 \
  --chunk-takeover-steps 1 \
  --visualize
```

可分别传入 `--sync-inference`、`--fixed-noise`、`--no-temporal-ensemble`、
`--accel-limit` 或 `--no-gripper-filter` 做消融对比。`--accel-limit` 不是程序
默认参数，但在当前chunk16香蕉抓取真机实验中已验证能明显消除关节抖动。不要为夹爪抖动场景传
`--gripper-latest-action`，该参数会让夹爪绕过时间集合结果。

关节抖动相关参数：

- `--noise-correlation 0.7`：相邻两次SmolVLA Flow Matching推理所用随机噪声的
  相关系数，取值范围为 `[0, 1)`。它只改变模型生成动作chunk时的随机采样连续性，
  不会向真实机械臂关节命令直接添加噪声。随机噪声按下式更新：

  ```text
  noise_new = ρ × noise_previous + sqrt(1 - ρ²) × innovation
  ```

  其中 `ρ` 是 `--noise-correlation`，`innovation` 是新的标准正态随机噪声。
  `sqrt(1-ρ²)` 用来保持合成噪声的总体方差基本不变，因此不能简单理解成
  “70%旧噪声加30%新噪声”。

  - `0`：每次推理完全独立采样，动作多样性最高，但相邻chunk更容易抖动。
  - `0.3～0.5`：保留较强的新随机性，适合动作过于迟钝或重复时对照。
  - `0.7`：当前默认折中值，通常能改善相邻chunk的一致性。
  - `0.8～0.9`：输出通常更稳定，但可能降低策略对新观测的响应灵活性。
  - 越接近 `1`：相邻推理噪声越相似；参数不能等于 `1`。

  该参数只在默认的随机噪声模式下生效。使用 `--fixed-noise` 后，每次推理都复用
  同一份固定噪声，`--noise-correlation` 不再参与噪声更新。它也不能解决GPU推理
  速度不足、动作chunk耗尽或机械臂执行层跟踪误差等问题。
- `--chunk-takeover-steps 1`：当前真机实测推荐值。只平滑新chunk第一个关节
  目标，能够减轻边界抖动，同时从第2步开始完全恢复模型动作。
- `--chunk-takeover-steps 2`：程序默认值，平滑前两个关节目标，但当前香蕉抓取
  任务实测不如 `1` 灵活。
- `--chunk-takeover-steps 0`：完全关闭短接管，用于异步基线对比。
- `--fixed-noise`：完全固定噪声，仅用于对比，不建议作为默认抓取配置。
- `--accel-limit`：启用最终关节位置命令的加速度连续整形。它不修改VLA的观测、
  推理结果或任务决策，也不是关节阻抗控制；VLA仍然提供目标关节动作，执行层只限制
  相邻控制周期之间的命令速度变化，抑制高频反向跳变。
- `--max-accel J1 J2 J3 J4 J5 J6`：分别设置六个关节的最大命令加速度，单位为
  `rad/s²`，仅在启用 `--accel-limit` 时生效。数值越小越平滑，但机械臂跟随VLA
  目标的延迟越大；数值越大响应越快，但抑制抖动的效果越弱。当前真机验证值为
  `2.5 2.5 2.5 3.5 3.5 3.5`。

当前 `chunk_size=16、n_action_steps=16` 香蕉抓取模型的普通基线为：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana_chunk16/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --execute \
  --inference-every 1 \
  --rate 50 \
  --noise-correlation 0.7 \
  --chunk-takeover-steps 1 \
  --visualize \
  --yes
```

如果上述参数的时间集合平滑效果不明显，当前chunk16香蕉抓取任务真机验证的
“流畅运行推荐组”为：

```bash
python -m rebot_smolvla_real.workflow.deploy_new \
  --checkpoint rebot_smolvla_real/ckpt/smolvla_rebot_real_banana_chunk16/checkpoints/last/pretrained_model \
  --instruction "抓取香蕉并放置到盘子中" \
  --execute \
  --inference-every 1 \
  --rate 50 \
  --noise-correlation 0.9 \
  --chunk-takeover-steps 2 \
  --temporal-ensemble \
  --temporal-ensemble-k 0.05 \
  --temporal-ensemble-history 5 \
  --accel-limit \
  --max-accel 2.5 2.5 2.5 3.5 3.5 3.5 \
  --visualize \
  --yes
```

这组参数仍然只使用 `deploy_new.py` 已有机制，不增加阻抗控制或轨迹引导：

- `--noise-correlation 0.9` 提高相邻推理随机噪声的一致性；
- `--chunk-takeover-steps 2` 平滑新动作块的前两个关节目标；
- `--temporal-ensemble-k 0.05` 相比默认 `0.2` 更均衡地融合新旧预测，平滑更强；
- `--temporal-ensemble-history 5` 已足够容纳当前实际最多约两个重叠预测；
- `--accel-limit --max-accel 2.5 2.5 2.5 3.5 3.5 3.5` 对时间集合无法覆盖的
  `ensemble=1` 区间继续约束关节命令的加速度。当前真机结果表现为关节不再抖动，
  且整段抓取运动流畅。

应先运行普通基线，再在相同初始姿态和物体位置下运行更强平滑组。终端中的
`ensemble` 是当前控制步实际融合的预测数：持续为 `1` 表示该时刻没有可融合的
重叠预测，此时仅调小 `temporal-ensemble-k` 不会产生明显效果。`chunk_size=16`
在50Hz下覆盖320ms；如果单次推理接近或超过该时间，能同时覆盖当前控制步的动作块
仍会很少。若更强参数导致抓取响应迟缓或夹爪时机变差，按
提高 `max-accel`、`temporal-ensemble-k 0.2`、`chunk-takeover-steps 1`、
`noise-correlation 0.7` 的顺序逐项放宽或退回普通基线，避免一次修改多个变量后
无法判断原因。若仍然抖动，可小幅降低 `max-accel`，但过低会让机械臂落后于VLA
动作时间轴，可能造成机械臂尚未到位而夹爪已经闭合。

## 8. 安全边界

采集入口会真实下发主手机械臂和夹爪命令。启动前必须确认工作空间无人员和障碍物、
急停可用，并先用短 episode 验证方向、限位和夹爪范围。

首次SmolVLA真机部署应满足：

- 工作空间内无人员；
- 有物理急停；
- 低刚度或低速度；
- 限制单步动作；
- 机械臂从训练分布内的初始姿态启动；
- 操作者可以立即终止控制循环。
