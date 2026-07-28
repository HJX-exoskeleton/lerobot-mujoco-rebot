#!/bin/bash
set -e

deactivate 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME
unset VIRTUAL_ENV

source /home/anaconda3/etc/profile.d/conda.sh
conda activate lerobot_rebot

cd /home/hjx/hjx_file/rebot_devarm_ws/rebotArm_policy_learning/act/rebot_scripts

python3 Servo_control/home_servo.py

python3 Servo_control/home_rebot.py

