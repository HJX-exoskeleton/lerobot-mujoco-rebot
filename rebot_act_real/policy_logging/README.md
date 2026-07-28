# ACT 真机推理实验记录

普通 ACT 和 MIT 阻抗部署默认不记录。只有在部署命令中显式加入 `--record` 后，
程序才会在 `rebot_act_real/policy_logging/runs/` 下为本次运行创建唯一目录。
每个实验目录包含：

- `metadata.json`：完整命令行、checkpoint/数据集参数、Git commit、模态、单位和时钟约定；
- `summary.json`：时长、样本数、丢帧数、磁盘占用及完成状态；
- `data/chunk_*.npz`：崩溃安全的分块列式数据；
- `videos/cam_{high,wrist}_NNNNNN.mp4`：仅在使用 `--record-images` 时产生；
- `figures/*.png`：由分析命令生成的300dpi论文图；
- `figures/metrics.json`：推理时延分位数、控制抖动、超时帧数和逐关节跟踪误差统计。
- `figures/keyframes.png`：双相机同步关键帧拼图；
- `figures/overview.png`：根据可用模态动态排版的论文级综合总览大图；
- `figures/tactile_analysis.png`：仅在存在触觉数据时生成的触觉时空分析大图；
- `figures/imu_analysis.png`：仅在存在IMU数据时生成的三维姿态与动力学分析大图；
- `figures/keyframes.json`：关键帧控制步、时间、选帧依据及事件指标值。

核心字段包括关节反馈、估算速度、ACT 原始输出、经过限幅/滤波后的实际命令、
跟踪误差、推理序号和时延、动作年龄、控制周期、超时计数、相机帧号与采样时间，
以及启用时的 IMU、12×30 触觉阵列和 MIT 重力/力矩前馈量。时间统一使用相对本次
运行起点的单调时钟，`metadata.json` 同时保存 Unix 时间原点，便于跨设备对齐。

生成论文图：

```bash
python -m rebot_act_real.policy_logging.analyze \
  --run rebot_act_real/policy_logging/runs/20260728_120000_banana
```

启用 `--record-images` 的实验还会自动分析视频，选择任务开始/结束、夹爪动作变化、
机械臂动作变化、触觉峰值和最大跟踪误差等代表时刻，生成双相机对齐的
`keyframes.png`。相近事件会去重，并使用任务进程中的均匀帧补足拼图。

`overview.png` 在一张300dpi大图中组合双相机关键帧、关节反馈/命令、跟踪误差、
夹爪动作、推理和控制时序，并按数据实际存在情况加入IMU动态、触觉响应及触觉峰值
热力图。没有记录的视频、IMU或触觉不会显示，也不会保留空白模态面板。分析器仅
生成PNG；再次分析旧实验时会清理自己以前生成的同名PDF结果。

触觉实验会额外生成 `tactile_analysis.png`，其中包含接触最大值/均值和自适应阈值、
活跃taxel面积、接触压力中心、四个关键时刻的12×30阵列快照、峰值三维接触表面、
列方向时空响应图、接触中心轨迹，以及存在视频时与触觉峰值严格同步的双相机画面。
`metrics.json` 同时记录接触持续时间、峰值时刻、最大接触面积、峰值压力中心和
压力中心轨迹长度等定量指标。

IMU实验会额外生成 `imu_analysis.png`，联合展示四元数恢复的三维单位球姿态轨迹、
四元数与相对初始姿态旋转角、三轴角速度/加速度及模长、多个关键时刻的三维姿态
坐标架、角速度功率谱、角加速度/jerk运动强度，以及存在视频时与角速度峰值同步的
双相机画面。`metrics.json` 同时记录四元数归一化误差、角速度峰值与RMS、加速度
峰值、相对1g最大偏差、最大姿态变化和主导运动频率。

正式实验建议显式使用 `--record --run-name task_method_seedXX`。默认不保存视频以
保护 50 Hz 控制实时性，需要定性案例视频时使用 `--record --record-images`。
`--record-images` 不能脱离 `--record` 单独使用。NPZ 是原始不可变记录，后续统计和
绘图应输出到 `figures/` 或新的派生目录，不要覆盖原始分块。

视频按 `--record-chunk-size` 分段（默认每250个控制帧约一段）并在编码关闭后原子
提交。即使进程被强制终止，已经提交的分段仍可独立播放；最多丢失当前尚未提交的
最后一段。以 `.partial.mp4` 开头的隐藏文件表示中断时未完成的分段，不应作为实验
视频使用。

论文实验建议同时使用固定控制步数，例如
`--record --max-steps 800 --run-name task_method_seed01`。`--max-steps` 统计控制
循环而不是墙钟秒数；50 Hz 下800步名义上约16秒。达到步数后部署程序会正常退出，
刷新最后一个数据分块和视频分段，因此不需要手动按 Ctrl+C。若控制循环发生超时，
实际运行时长会略长，应使用日志中的 `duration_s` 和 `loop_dt_s` 报告真实时序。
