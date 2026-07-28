### Task parameters

DATA_DIR ='/media/hjx/PSSD/hjx_ws/data/rebot/data_real_tactile'

TASK_CONFIGS = {

    'rebot_real_grasp_banana': {
        'dataset_dir': DATA_DIR + '/rebot_real_grasp_banana',
        'num_episodes': 20,
        'episode_len': 800,
        'camera_names': ['cam_high', 'cam_wrist']
    },

}

