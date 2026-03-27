#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path

# get current workspace
current_file = Path(__file__)

import sys
parent_dir = current_file.parent
sys.path.append(str(parent_dir))

import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

class PI0_DUAL:
    def __init__(self, model_path, task_name, **kwargs):
        self.task_name = task_name

        train_config_name = kwargs.get("train_config_name", "pi05_aloha")

        print("train_config_name:", train_config_name)

        config = _config.get_config(train_config_name)
        print("get config success!")
        norm_stats = _normalize.load(Path(model_path) / "assets" / "1118")
        self.policy = _policy_config.create_trained_policy(config, model_path, norm_stats=norm_stats)
        print("loading model success!")
        self.img_size = (224,224)
        self.observation_window = None
        self.instruction = None
        # self.random_set_language()

    # set img_size
    def set_img_size(self,img_size):
        self.img_size = img_size
    
    # set language randomly
    def random_set_language(self):
        instruction = "Clamp bag, open it, put wire, earphone and manual in, push manual, pinch-seal bag gently."
        self.instruction = instruction
        print(f"successfully set instruction:{self.instruction}")
    
    def set_instruction(self, inst):
        self.instruction = inst
        print(f"successfully set instruction:{self.instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, instruction=None):
        if self.instruction is None:
            self.random_set_language()
        
        # imgs_array = []
        # for data in img_arr:
        #     jpeg_bytes = np.array(data).tobytes().rstrip(b"\0")
        #     nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        #     imgs_array.append(cv2.imdecode(nparr, 1))
        
        imgs_array = img_arr
        
        img_front, img_right, img_left = imgs_array[0], imgs_array[1], imgs_array[2]
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction if instruction is None else instruction,
        }
        # print(state)

    def get_action(self, **kwargs):
        assert (self.observation_window is not None), "update observation_window first!"
        return self.policy.infer(self.observation_window, **kwargs)["actions"] 

    def get_action_batch(self, obs_batch, **kwargs):
        return self.policy.infer(obs_batch, **kwargs)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
    
    def get_device(self):
        return self.policy._pytorch_device
if __name__ == "__main__":
    model_path = "/home/xspark-ai/project/openpi_ckpts/pi05/pytorch/rtc_pi05/"
    model = PI0_DUAL(model_path, "test")
    print("Model initialized!")

    print("\nTesting batch inference...")
    batch_size = 2
    obs_batch = {
        "state": np.random.randn(batch_size, 14).astype(np.float32),
        "images": {
            "cam_high": np.random.randint(0, 256, (batch_size, 3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(0, 256, (batch_size, 3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(0, 256, (batch_size, 3, 224, 224), dtype=np.uint8),
        },
        "prompt": ["pick up the block"] * batch_size
    }

    try:
        actions = model.get_action_batch(obs_batch)
        print(f"Batch inference success! Actions shape: {actions.shape}")
        if actions.shape[0] == batch_size:
            print(f"Verified: Output batch size matches input batch size ({batch_size}).")
    except Exception as e:
        print(f"Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
