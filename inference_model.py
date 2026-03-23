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
        train_config_name = kwargs.get("train_config_name", "pi05_full_base") 
        
        print(f"Use config: {train_config_name}")
        config = _config.get_config(train_config_name)
        print("get config success!")
        self.policy = _policy_config.create_trained_policy(config, model_path)
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
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_Path = os.path.join(root_dir, "task_instructions", f"{self.task_name}.json")
        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict['instructions']
        instruction = np.random.choice(instructions)
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")
    
    def set_instruction(self, inst):
        self.instruction = inst
        print(f"successfully set instruction:{self.instruction}")
    
    # Update the observation window buffer
    def update_observation_window(self, img_arr, state, instruction=None):
        # imgs_array = img_arr
        if self.instruction is None:
            self.random_set_language()
        imgs_array = []

        for data in img_arr:
            jpeg_bytes = np.array(data).tobytes().rstrip(b"\0")
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            imgs_array.append(cv2.imdecode(nparr, 1))
        # import pdb;pdb.set_trace()

        img_front, img_right, img_left = imgs_array[0][:,:,::-1], imgs_array[1][:,:,::-1], imgs_array[2][:,:,::-1]

        # cv2.imwrite("head.jpg", img_front)
        # cv2.imwrite("left.jpg", img_left)
        # cv2.imwrite("right.jpg", img_right)

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

    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

if __name__ == "__main__":
    model = PI0_DUAL("/home/xspark-ai/project/openpi/checkpoint/pi05/pytorch/rtc_pi05/","test", False, False)
    print("succ!")
    