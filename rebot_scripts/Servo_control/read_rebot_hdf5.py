import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

path = '/media/hjx/PSSD/hjx_ws/data/rebot/data_real/rebot_real_grasp_banana/episode_0.hdf5'

obj = h5py.File(path)
print(obj.keys())
print(obj['action'])
print(obj['action'].keys())
print(obj['action']['target_pos'])
print('-------------------------')
print(obj['observations'])
print(obj['observations'].keys())
print('-------------------------')
print(obj['observations']['qpos'])
print(obj['observations']['qvel'])
print('-------------------------')

print(obj['observations']['images'].keys())
print(obj['observations']['images']['cam_high'])
print(obj['observations']['images']['cam_wrist'])
print('-------------------------')



### 检查 HDF5 每个数据集大小
def fmt_size(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

print("文件路径:", path)
print("文件总大小:", fmt_size(os.path.getsize(path)))
print()

with h5py.File(path, "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            logical_size = obj.size * obj.dtype.itemsize
            disk_size = obj.id.get_storage_size()

            print(f"{name}")
            print(f"  shape       = {obj.shape}")
            print(f"  dtype       = {obj.dtype}")
            print(f"  logical     = {fmt_size(logical_size)}")
            print(f"  disk        = {fmt_size(disk_size)}")
            print(f"  compression = {obj.compression}")
            print()
    f.visititems(visit)




### 检查相机真实新帧数
with h5py.File(path, "r") as f:
    for cam in ["cam_high", "cam_wrist"]:
        ids = f[f"/observations/image_frame_ids/{cam}"][:]

        unique_count = len(np.unique(ids))
        total_count = len(ids)
        repeat_ratio = 1.0 - unique_count / total_count

        print(cam)
        print("  保存帧数:", total_count)
        print("  真实相机新帧数:", unique_count)
        print("  重复率:", f"{repeat_ratio * 100:.1f}%")
        print("  起始 frame_id:", ids[0])
        print("  结束 frame_id:", ids[-1])
        print()

