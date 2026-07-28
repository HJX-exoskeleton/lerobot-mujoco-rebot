#!/usr/bin/env python3

"""
专业级 IMU 数据可视化仪表盘 (支持 EMA 实时滤波)
请确保已安装依赖: pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from YbImuLib import YbImuSerial

# 串口设备路径，请替换为实际端口
SERIAL_PORT = "/dev/ttyUSB1"  # 惯性测量单元 IMU 串口

class IMUDashboard:
    def __init__(self, port, max_points=100, interval=50, alpha=0.15):
        """
        初始化仪表盘
        :param port: 串口路径
        :param max_points: 屏幕上X轴最多显示的数据帧数
        :param interval: 图表刷新间隔 (毫秒)
        :param alpha: EMA 滤波系数 (0~1)。越接近0越平滑但有延迟，越接近1越灵敏但有毛刺。
        """
        self.port = port
        self.max_points = max_points
        self.interval = interval
        self.alpha = alpha  
        
        self.last_filtered_data = None # 用于保存上一帧的平滑结果，供滤波计算使用
        
        # 1. 初始化硬件连接
        print("正在连接 IMU 硬件...")
        self.imu = YbImuSerial(self.port, debug=False)
        self.imu.create_receive_threading()
        print("IMU 连接成功！")
        
        # 2. 初始化数据缓冲区
        self.x_data = deque([0]*max_points, maxlen=max_points)
        self.frame_count = 0
        
        # 使用字典结构严密分组管理双端队列
        self.data_buffers = {
            'accel': {axis: deque([0]*max_points, maxlen=max_points) for axis in ['x', 'y', 'z']},
            'gyro':  {axis: deque([0]*max_points, maxlen=max_points) for axis in ['x', 'y', 'z']},
            'mag':   {axis: deque([0]*max_points, maxlen=max_points) for axis in ['x', 'y', 'z']},
            'euler': {axis: deque([0]*max_points, maxlen=max_points) for axis in ['roll', 'pitch', 'yaw']},
            'quat':  {axis: deque([0]*max_points, maxlen=max_points) for axis in ['w', 'x', 'y', 'z']},
            'baro':  {'height': deque([0]*max_points, maxlen=max_points)}
        }
        
        # 3. 构建 UI 界面
        self._setup_figure()

    def _setup_figure(self):
        """配置 3x2 布局的 Matplotlib 仪表盘"""
        plt.style.use('fast') 
        self.fig, self.axs = plt.subplots(3, 2, figsize=(14, 8))
        self.fig.suptitle('IMU Real-Time Sensor Dashboard (Filtered)', fontsize=16, fontweight='bold')
        
        # 定义每个子图的配置字典
        self.plot_configs = [
            (self.axs[0, 0], 'accel', 'Accelerometer (g)', ['x', 'y', 'z'], ['red', 'green', 'blue']),
            (self.axs[0, 1], 'gyro', 'Gyroscope (rad/s)', ['x', 'y', 'z'], ['red', 'green', 'blue']),
            (self.axs[1, 0], 'mag', 'Magnetometer (uT)', ['x', 'y', 'z'], ['red', 'green', 'blue']),
            (self.axs[1, 1], 'euler', 'Euler Angles (deg)', ['roll', 'pitch', 'yaw'], ['red', 'green', 'blue']),
            (self.axs[2, 0], 'quat', 'Quaternion', ['w', 'x', 'y', 'z'], ['black', 'red', 'green', 'blue']),
            (self.axs[2, 1], 'baro', 'Barometer Altitude (m)', ['height'], ['purple'])
        ]
        
        self.lines = {} # 存储所有线条对象
        
        for ax, key, title, labels, colors in self.plot_configs:
            ax.set_title(title, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            self.lines[key] = {}
            for label, color in zip(labels, colors):
                line, = ax.plot([], [], label=label, color=color, linewidth=1.5)
                self.lines[key][label] = line
            ax.legend(loc='upper left', fontsize=8)

        # 调整布局，防止标签重叠
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def fetch_new_data(self):
        """从传感器读取最新的一帧数据并进行 EMA 滤波平滑"""
        try:
            # 拉取硬件原始数据
            ax, ay, az = self.imu.get_accelerometer_data()
            gx, gy, gz = self.imu.get_gyroscope_data()
            mx, my, mz = self.imu.get_magnetometer_data()
            roll, pitch, yaw = self.imu.get_imu_attitude_data(ToAngle=True)
            qw, qx, qy, qz = self.imu.get_imu_quaternion_data()
            height, temp, press, press_diff = self.imu.get_baro_data()

            # 构建当前帧的原始数据字典
            raw_data = {
                'accel': {'x': ax, 'y': ay, 'z': az},
                'gyro':  {'x': gx, 'y': gy, 'z': gz},
                'mag':   {'x': mx, 'y': my, 'z': mz},
                'euler': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'quat':  {'w': qw, 'x': qx, 'y': qy, 'z': qz},
                'baro':  {'height': height}
            }

            # 如果是第一帧数据，直接作为历史基准，不进行滤波
            if self.last_filtered_data is None:
                self.last_filtered_data = raw_data
                return raw_data

            # 遍历所有数据，应用 EMA 滤波公式
            filtered_data = {}
            for category, axes in raw_data.items():
                filtered_data[category] = {}
                for axis, raw_val in axes.items():
                    prev_val = self.last_filtered_data[category][axis]
                    
                    # 核心滤波公式：Y_t = alpha * X_t + (1 - alpha) * Y_t-1
                    smoothed_val = self.alpha * raw_val + (1 - self.alpha) * prev_val
                    
                    filtered_data[category][axis] = smoothed_val
                    
                    # 更新历史状态，供下一帧使用
                    self.last_filtered_data[category][axis] = smoothed_val

            return filtered_data

        except Exception as e:
            print(f"Data read error: {e}")
            return None

    def update(self, frame):
        """动画更新回调函数"""
        new_data = self.fetch_new_data()
        if not new_data:
            return []

        # 更新 X 轴时间帧计数
        self.frame_count += 1
        self.x_data.append(self.frame_count)

        updated_lines = []

        # 遍历所有传感器数据组，更新数据队列和图表
        for ax, key, _, labels, _ in self.plot_configs:
            all_y_values_in_plot = []
            
            for label in labels:
                # 1. 提取滤波后的数据并入队
                val = new_data[key][label]
                self.data_buffers[key][label].append(val)
                
                # 2. 更新线条数据
                line = self.lines[key][label]
                line.set_data(self.x_data, self.data_buffers[key][label])
                updated_lines.append(line)
                
                all_y_values_in_plot.extend(self.data_buffers[key][label])

            # 3. 动态调整当前子图的 X 轴和 Y 轴范围
            ax.set_xlim(self.x_data[0], self.x_data[-1])
            
            # 获取当前数据的最大最小值并留出 10% 的余量，防止波形溢出屏幕或剧烈抖动
            if all_y_values_in_plot:
                y_min, y_max = min(all_y_values_in_plot), max(all_y_values_in_plot)
                margin = max(abs(y_max - y_min) * 0.1, 0.01) 
                ax.set_ylim(y_min - margin, y_max + margin)

        return updated_lines

    def run(self):
        """启动动画主循环"""
        print("正在启动数据可视化仪表盘 (请在弹出的窗口中查看)...")
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            interval=self.interval, 
            blit=False, 
            cache_frame_data=False
        )
        # 显示窗口 (阻塞主线程)
        plt.show()

if __name__ == "__main__":
    # 初始化仪表盘，可在此处调整滤波系数 alpha (推荐值 0.1 ~ 0.3)
    dashboard = IMUDashboard(port=SERIAL_PORT, max_points=100, interval=50, alpha=0.15)
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n程序已手动终止。")
