# LeRobot MuJoCo 项目理解与命令行复现

## 1. 项目在做什么

项目用 MuJoCo 中的 ROBOTIS OMY 机械臂完成桌面抓放。观测包含固定相机图像、腕部相机图像和 6 维状态，动作为 6 个关节角加 1 个夹爪量。代码分成两条流水线：

1. **单任务流水线**：键盘遥操作采集“把杯子放到盘子上”的数据，训练 ACT，再在仿真中闭环部署。
2. **语言条件流水线**：随机组合物体和自然语言指令，采集数据，微调 Pi0 或 SmolVLA，再在仿真中根据指令闭环部署。

`mujoco_env/` 负责仿真、渲染、键盘控制、逆运动学和成功判定；`asset/` 是机器人、桌面和物体模型；LeRobot 负责数据格式、模型和训练基础设施。

## 2. Notebook 与 Python 脚本对应关系

| 原 Notebook | 终端脚本 | 功能 |
|---|---|---|
| `1.collect_data.ipynb` | `scripts/01_collect_data.py` | 采集单任务演示 |
| `2.visualize_data.ipynb` | `scripts/02_visualize_data.py` | 回放单任务数据 |
| `3.train.ipynb` | `scripts/03_train_act.py` | 训练并评估 ACT |
| `4.deploy.ipynb` | `scripts/04_deploy_act.py` | 部署 ACT |
| `5.language_env.ipynb` | `scripts/05_collect_language_data.py` | 采集语言条件数据 |
| `6.visualize_data.ipynb` | `scripts/06_visualize_language_data.py` | 回放/上传语言数据 |
| `7.pi0.ipynb` | `scripts/07_pi0.py` | 训练或部署 Pi0 |
| `8.smolvla.ipynb` | `scripts/08_smolvla.py` | 训练或部署 SmolVLA |

所有命令都应在项目根目录执行，建议使用 `python -m scripts.<模块名>`，这样项目根目录会正确加入 Python 搜索路径。

## 3. 完成环境安装

先按照 `commands.txt` 安装环境。若环境是用旧版文件安装的，必须先隔离用户目录并补齐依赖：

```bash
conda activate lerobot_rebot
conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=""
conda deactivate
conda activate lerobot_rebot

python -m pip uninstall -y opencv-python-headless opencv-python
python -m pip install --force-reinstall -c constraints-rebot.txt \
  "numpy==1.26.4" "opencv-python>=4.9,<4.12"
python -m pip install -r requirements-rebot.txt
python -m pip install --upgrade -c constraints-rebot.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "lerobot[pi0,smolvla] @ https://github.com/huggingface/lerobot/archive/10b7b3532543b4adfb65760f02a49b4c537afde7.zip"
# LeRobot 的包元数据会再次拉取 headless 版；真机环境必须在最后移除它，
# 并强制恢复带 Qt/GTK 窗口支持的 cv2 文件。
python -m pip uninstall -y opencv-python-headless
python -m pip install --force-reinstall --no-deps -c constraints-rebot.txt \
  "opencv-python>=4.9,<4.12"
python -c "import cv2; assert 'GUI:                           NONE' not in cv2.getBuildInformation(); print('OpenCV GUI: OK')"
python -m pip check
```

说明：LeRobot 当前包元数据声明依赖 `opencv-python-headless`，因此最后的
`pip check` 可能报告 LeRobot 缺少该发行包。这是已知的包元数据差异；项目实际
使用兼容的 `opencv-python` 提供同一个 `cv2` API，以支持真机相机预览。不要在
同一环境同时安装 GUI 与 headless 两个 OpenCV wheel。

### 3.1 Astra-S / OpenNI2

`requirements-rebot.txt` 已固定安装 Astra-S 脚本使用的 Python 绑定：

```bash
python -m pip install -c constraints-rebot.txt "openni==2.3.0"
python -c "from openni import openni2; print('OpenNI Python binding: OK')"
```

注意，`openni==2.3.0` 只提供 Python 绑定，不包含 Orbbec 的原生 OpenNI2
驱动和动态库。当前 Astra-S 脚本默认使用：

```text
/home/hjx/orbbec_openni_redist
```

该路径应指向 Orbbec OpenNI2 SDK 的 `sdk/libs` 目录。可以执行以下检查：

```bash
test -d /home/hjx/orbbec_openni_redist
python - <<'PY'
from openni import openni2

redist = "/home/hjx/orbbec_openni_redist"
openni2.initialize(redist)
print("OpenNI devices:", openni2.Device.enumerate_uris())
PY
```

`primesense` 不是当前脚本的必要依赖；项目统一使用
`from openni import openni2`。

### 3.2 RealSense D405

`requirements-rebot.txt` 已固定安装 D405 相机使用的 RealSense Python
SDK：

```bash
python -m pip install -c constraints-rebot.txt \
  "pyrealsense2==2.58.3.10794"
python -c "import pyrealsense2 as rs; print('RealSense Python binding: OK')"
```

连接 D405 后，可以使用项目中的预览脚本验证彩色图像：

```bash
python rebot_scripts/CameraPreview_D405_test.py
```

`pyrealsense2` 提供 Python 接口和对应的本地扩展，但 Linux 主机仍需具备
USB 设备访问权限。若模块能够导入但无法枚举设备，应继续检查 RealSense
udev 规则、USB 连接以及设备是否被其他进程占用。

### 3.3 reBot 电机通信

真机机械臂和夹爪控制代码使用 `motorbridge` 的 Rust ABI Python SDK。
`requirements-rebot.txt` 已固定当前验证版本：

```bash
python -m pip install -c constraints-rebot.txt "motorbridge==0.5.0"
python -c "from motorbridge import Controller, Mode, CallError; print('motorbridge: OK')"
```

能够导入 `motorbridge` 只表示 SDK 安装成功，并不代表电机通信已经建立。
连接真机前还需要确认对应 CAN/串口设备存在、当前用户具有设备访问权限，并且
没有其他控制进程占用通信设备。机械臂上电后的使能、回零和动作测试应按照真机
安全流程执行。

### 3.4 Pinocchio 运动学与动力学

`reBotArm_control_py` 的运动学、动力学和轨迹模块使用 Pinocchio。PyPI
发行包名称是 `pin`，Python 导入模块名称则是 `pinocchio`：

```bash
python -m pip install -c constraints-rebot.txt "pin==2.6.2"
python -c "import pinocchio as pin; print('Pinocchio:', pin.__version__)"
```

不要使用 `import pin` 验证安装，因为该发行包不提供名为 `pin` 的顶层模块。

如果导入错误中出现类似：

```text
libboost_python38.so
undefined symbol: _Py_fopen
```

说明 Python 3.10 环境错误加载了系统中的 Python 3.8 Boost.Python 动态库，
并非代码缺少 `pinocchio`。此时应先检查 `LD_LIBRARY_PATH` 和 Pinocchio
动态库解析结果，不应在导入失败的状态下进行运动学或动力学实验。

解压场景资源（只需一次）：

```bash
unzip asset/objaverse/plate_11.zip -d asset/objaverse
```

## 4. 复现路线 A：ACT

### 4.1 采集数据

```bash
python -m scripts.01_collect_data --episodes 20 --root demo_data
```

键盘控制：`W/A/S/D` 控制 XY，`R/F` 控制 Z，`Q/E` 和方向键控制姿态，空格切换夹爪，`Z` 丢弃当前 episode 并重置。机械臂开始移动后才记录；满足成功条件后自动保存 episode。

如果要删除旧数据并重新采集，必须显式添加 `--overwrite`：

```bash
python -m scripts.01_collect_data --episodes 20 --overwrite
```

### 4.2 回放检查

```bash
python -m scripts.02_visualize_data --root demo_data --episode 0
```

### 4.3 训练 ACT

先做 20 步冒烟测试：

```bash
python -m scripts.03_train_act --steps 20 --batch-size 2 --output ckpt/act_smoke
```

确认无误后正式训练：

```bash
python -m scripts.03_train_act --steps 5000 --batch-size 8 --output ckpt/act_y
```

如果某些 episode 质量不好，不要直接删除 Parquet 文件。例如排除 episode 0 与 episode 1，将评估 episode 改成未排除的编号，例如 2：

```bash
python -m scripts.03_train_act --exclude-episodes 0 1 --eval-episode 2 --steps 5000 --batch-size 8 --output ckpt/act_y
```

可一次排除多个 episode，例如 `--exclude-episodes 0 3 7`。被排除的数据既不会进入 DataLoader，也不会参与模型归一化统计量，原始数据仍保留，便于反悔或重新检查。

  - episode 0、1：不参与训练和统计
  - episode 2：参与训练，并在训练结束后用于动作误差评估
  - episode 2–21：参与训练


脚本会保存 checkpoint 和 `action_error.png`。显存不足时减小 `--batch-size`。

### 4.4 部署 ACT

```bash
python -m scripts.04_deploy_act --checkpoint ckpt/act_y
```

## 5. 复现路线 B：Pi0/SmolVLA

### 5.1 获取语言数据

自行采集：

```bash
python -m scripts.05_collect_language_data --episodes 20 --root demo_data_language
```

重新采集并覆盖原来的语言条件数据集，执行：

```bash
python -m scripts.05_collect_language_data --episodes 20 --root demo_data_language --overwrite
```

  其中：

  - --episodes 20：采集 20 条 episode。
  - --root demo_data_language：数据保存目录。
  - --overwrite：先删除原数据集，再创建新数据集。


或者下载作者的数据集：

```bash
huggingface-cli download Jeongeun/omy_pnp_language \
  --repo-type dataset --local-dir demo_data_language
```

回放检查：

```bash
python -m scripts.06_visualize_language_data --root demo_data_language --episode 0
```

### 5.2 训练 SmolVLA（建议先做）

`smolvla_omy.yaml` 已调整为 RTX 4060 8 GB 的保守起点：batch size 1、worker 0、关闭 WandB。

先把 YAML 中的 `steps` 临时改成 `20` 做冒烟测试，然后执行：

```bash
python -m scripts.08_smolvla train
```

训练会优先读取项目内的预训练权重缓存：

```text
models/models--lerobot--smolvla_base/
```

代码会通过 `refs/main` 自动解析具体 snapshot。也可以显式指定缓存根目录或 snapshot：

```bash
python -m scripts.08_smolvla train \
  --pretrained models/models--lerobot--smolvla_base
```

只有本地目录不存在时才会回退到 `lerobot/smolvla_base` 在线下载。

正式训练时再将 `steps` 调回所需值。首次运行会从 Hugging Face 下载基础模型，需要联网并占用较大磁盘空间。

终端训练进度条会实时显示：完成比例、step、ETA、loss、学习率、单步耗时、样本吞吐率和 CUDA 峰值显存。例如：

```text
Training smolvla: 12%|...| 2400/20000 [16:48:00<123:20:00, loss=0.0470, lr=9.90e-05, step_s=25.20, sample_s=0.63, gpu_GB=5.10]
```

部署本地 checkpoint：

```bash
python -m scripts.08_smolvla deploy
```

直接部署作者的 Hub 模型：

```bash
python -m scripts.08_smolvla deploy --hub-model Jeongeun/omy_pnp_smolvla
```

部署入口会在导入 Hugging Face 组件前将缓存设置到项目的 `models/`。作者模型会保存到：

```text
models/models--Jeongeun--omy_pnp_smolvla/
```

不会写入 `$HOME/.cache/huggingface`。

### 5.3 Pi0 授权、下载、推理与训练

Pi0 使用受限访问的 `google/paligemma-3b-pt-224` tokenizer/config。首次使用前必须完成以下授权流程。

1. 登录 Hugging Face 网站，打开 `https://huggingface.co/google/paligemma-3b-pt-224`，阅读并接受 Google 使用条款。访问申请只能在浏览器完成。
2. 在 `https://huggingface.co/settings/tokens` 创建一个具有 `Read` 权限的个人 token。不要把 token 写进脚本、Git 或聊天记录。
3. 在终端导出项目缓存目录。必须使用 `export`；仅写 `HF_HOME=...` 不会把变量传给 `hf` 子进程：

```bash
cd /media/hjx/新加卷/hjx_ws/lerobot-mujoco-tutorial
conda activate lerobot_rebot

export HF_HOME="$PWD/models/.hf_home"
export HF_HUB_CACHE="$PWD/models"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

检查变量：

```bash
env | grep -E '^HF_HOME=|^HF_HUB_CACHE='
```

登录并粘贴 Read token；询问是否加入 Git credential 时选择 `n`：

```bash
hf auth login
hf auth whoami
ls -l "$HF_HOME/token"
```

如果登录时出现临时 SSL EOF，先确认网络后重试，不要关闭 SSL 校验：

```bash
curl -I --retry 3 https://huggingface.co
hf auth login
```

直接下载并部署作者微调好的 Pi0：

```bash
python -m scripts.07_pi0 deploy --hub-model Jeongeun/omy_pnp_pi0
```

部署脚本会按需自动下载，不需要手动下载整个 PaliGemma 3B 仓库。缓存目录为：

```text
models/models--Jeongeun--omy_pnp_pi0/
models/models--google--paligemma-3b-pt-224/
models/.hf_home/token
```

其中 `Jeongeun/omy_pnp_pi0` 是 OMY 任务微调权重；PaliGemma 目录主要提供 Pi0 代码需要的 tokenizer/config。部署还需要 `demo_data_language/meta/` 中的数据特征和归一化统计量。

常见错误：

- `401 GatedRepoError`：尚未在网页接受 PaliGemma 条款、token 不属于已授权账户，或登录使用了错误的 `HF_HOME`。
- `SSL UNEXPECTED_EOF`：网络连接被中途断开，使用 `curl` 检查后重试。
- `Xet ConnectionError`：项目默认禁用 Xet 并改用普通 HTTP；再次执行同一命令会利用 `.incomplete` 文件续传。
- `CUDA out of memory`：Pi0 超过本机可用显存，与下载或 token 无关。

训练自己的 Pi0：

```bash
python -m scripts.07_pi0 train
```

会优先加载作者已经微调好的模型：

  models/models--Jeongeun--omy_pnp_pi0

  然后使用你的 demo_data_language 数据继续微调。加载顺序目前是：

  1. 作者模型 models--Jeongeun--omy_pnp_pi0
  2. 官方模型 models--lerobot--pi0
  3. 在线模型 lerobot/pi0


可以显式选择官方基础模型：

```bash
python -m scripts.07_pi0 train --pretrained lerobot/pi0
```

如果官方模型已完整下载，则可以使用本地缓存：

```bash
python -m scripts.07_pi0 train --pretrained models/models--lerobot--pi0
```

  两种路线的区别是：

  - 从 Jeongeun/omy_pnp_pi0 继续训练：属于二次微调，对相似的 OMY 抓取任务通常收敛更快。
  - 从 lerobot/pi0 开始训练：属于从官方通用基础模型进行任务微调，更接近项目原始训练流程，但需要更长训练时间。


Pi0 基础模型、tokenizer 和 processor 默认下载到项目目录，而不是占用 Home：

```text
models/models--lerobot--pi0/
```

其他由 Pi0 引用的 Hugging Face 模型也会写入 `models/models--<组织>--<模型>/`。如需改变缓存位置：

```bash
python -m scripts.07_pi0 train --cache-dir /其他磁盘/huggingface-models
```

`pi0_omy.yaml` 已设为 batch size 1、冻结视觉编码器、只训练 action expert。不过 Pi0 仍明显大于 SmolVLA；8 GB 显存可能在模型加载或反向传播时 OOM。出现 OOM 不表示环境错误，而是模型、优化器状态和激活值超过显存。此时优先使用更大显存 GPU、云端训练，或使用上面的作者 checkpoint。

本地训练成功后部署：

```bash
python -m scripts.07_pi0 deploy
```

也可以显式指定检查点：

```bash
python -m scripts.07_pi0 deploy --checkpoint ckpt/pi0_omy/checkpoints/last/pretrained_model
```

如果要推理某个具体训练节点，也可以使用：

```bash
python -m scripts.07_pi0 deploy --checkpoint ckpt/pi0_omy/checkpoints/010000/pretrained_model
```

## 6. 常见检查

查看每个脚本的参数：

```bash
python -m scripts.01_collect_data --help
python -m scripts.03_train_act --help
python -m scripts.07_pi0 --help
python -m scripts.08_smolvla --help
```

确认当前解释器和 GPU：

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

确认完整依赖来自当前环境：

```bash
python -m pip check
python -c "import numpy, pyarrow, glfw, mujoco, lerobot; print('runtime imports: OK')"
```

MuJoCo viewer 需要图形桌面和 `DISPLAY`，不能直接在无显示的纯 SSH 会话中打开。数据采集、回放和部署脚本会一直运行到 viewer 被关闭、采集数量完成或任务成功。
