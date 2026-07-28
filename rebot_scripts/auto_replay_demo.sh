#!/bin/bash
set -e

deactivate 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV

source /home/anaconda3/etc/profile.d/conda.sh
conda activate lerobot_rebot

cd /home/hjx/hjx_file/rebot_devarm_ws/rebotArm_policy_learning/act/rebot_scripts

python3 Servo_control/replay_rebot_episodes.py \
  --dataset /media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_test/episode_0.hdf5 \
  --no-final-disable \

#python3 Servo_control/replay_rebot_episodes.py \
#  --dataset /media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_grasp_banana/episode_0.hdf5 \
#  --no-final-disable \

python3 Servo_control/home_rebot.py

