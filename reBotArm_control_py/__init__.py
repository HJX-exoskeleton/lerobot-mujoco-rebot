"""reBotArm_control_py - reBotArm 机械臂 Python 控制库。"""
# from . import actuator
# from . import kinematics
# from . import dynamics
#
# __all__ = ["actuator", "kinematics", "dynamics"]


# new
# actuator 是真机控制必须模块，RobotArm 在这里面
from . import actuator

# kinematics 依赖 pinocchio。
# replay / home / pos_vel 真机控制不需要 kinematics，
# 所以这里改成可选导入，避免 pinocchio 环境问题导致 RobotArm 无法导入。
try:
    from . import kinematics
except Exception as e:
    kinematics = None
    print(f"[reBotArm_control_py] 跳过 kinematics 导入: {e}")

# dynamics 也可能依赖额外动力学/运动学库，也改成可选导入
try:
    from . import dynamics
except Exception as e:
    dynamics = None
    print(f"[reBotArm_control_py] 跳过 dynamics 导入: {e}")

__all__ = ["actuator", "kinematics", "dynamics"]

