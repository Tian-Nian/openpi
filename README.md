## 如何训练openpi模型
### step1. 转化你的数据集到lerobot格式:
1. 定义lerobot接收数据的格式:
``` python
dataset = LeRobotDataset.create(
    repo_id=repo_id,
    robot_type="realman", # 这个无关紧要
    fps=10,
    features={ 
        # 实际上你可以定义任何的数据结构,只要你索引对应数据方便就行
        # 因为模型在训练的时候是取索引对应路径下的数据的
        # 对于一个数据,你只需要把dtype和shape写对, names是补充说明用的
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": [
            "channels",
            "height",
            "width",
        ],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": [
            "channels",
            "height",
            "width",
        ],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": [
            "channels",
            "height",
            "width",
        ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": ["r1,r2,r3,r4,r5,r6,r7,gr,l1,l2,l3,l4,l5,l6,l7,gl"],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": ["r1,r2,r3,r4,r5,r6,r7,gr,l1,l2,l3,l4,l5,l6,l7,gl"],
        },
    },
    image_writer_threads=10,
    image_writer_processes=5,
)
```
2. 将你的数据填到注册的`dataset`下
``` python
# 读入你存储的数据, 按照每一帧进行读取, 例如你的数据有50帧, 那么num_frame = 50
for i in range(num_frames):
    frame = {
        "observation.state": state[i], # 对应的state, 维度要和你注册的时候一样!
        "action": action[i],
    }

    for camera, img_array in imgs_per_cam.items():
        if camera == "cam_head":
            frame["observation.images.cam_high"] = img_array[i]
        elif camera=="cam_left_wrist":
            frame["observation.images.cam_left_wrist"] = img_array[i]
        elif camera=="cam_right_wrist":
            frame["observation.images.cam_right_wrist"] = img_array[i]

    dataset.add_frame(frame) # 将当前保存好的对应帧加入
# 生成当前tarjectory对应episode, 由于不能存储string, 所以将你的指令写到这一条轨迹的名称里
dataset.save_episode(task=instruction) 
```
3. 写入lerobot数据
``` python
dataset.consolidate()
```
默认保存路径在:
``` bash
~/.cache/huggingface/lerobot
```
如果需要更改路径:
``` bash
export LEROBOT_HOME="your path"
```

### step2. 设置`config.py`:
我已经将我的config.py替换进去了, 提供了单臂与双臂两种类型, 单臂基于libero_policy,但是action和state是对齐维度(关节角控制),双臂采用aloha_policy, 取消了adapt_to_pi的设置.
这里拿出一个讲解下详细配置.
``` python
TrainConfig(
    name="pi0_base_aloha_lora", # 你的config_name, 后面训练会用到, 可以随意设置
    model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"), # 这是配置为lora的设置
    data=LeRobotAlohaDataConfig( # 使用Aloha格式来读取数据, 这个可以自己写
        # 要实现两个函数:policy.Input()定义数据怎么被输入 ,policy.Output() 定义输出格式, 要和你目标的action对齐维度, 没对齐不会报错但可能有问题
        repo_id="your repo id",# your datasets repo_id
        adapt_to_pi = False, # Alohapolicy才有这个设置, 建议设置为False
        repack_transforms=_transforms.Group(
            inputs=[
                # 左侧: Aloha_policy中的对于图像的二次索引, 将图像按顺序堆叠后输入模型用的, 不需要变
                # 右侧: 填写你lerobot格式对应数据索引
                _transforms.RepackTransform( 
                    { 
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        # 你自己的数据集当然是在本地的, 你如果设置了语言指令是在task name中那就要True
        base_config=DataConfig(
            local_files_only=True,  # Set to True for local-only datasets.
            prompt_from_task=True,  # Set to True for prompt by task_name
        ),
    ),
    # 表示这个属性不能更改的
    freeze_filter=pi0.Pi0Config(
        action_dim=16,paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    batch_size=32, # the total batch_size not pre_gpu batch_size
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"), # 你选择的索引模型, 可以是本地路径,就正常索引就行
    num_train_steps=30000,
    fsdp_devices=1, # 如果你单卡内存不够, 那就多卡喽, 具体看config.py line 359
),
```

### step3. 开启训练

``` python
# 计算数据的norm stat, 数据是按照你的config下索引的repo_id来的
uv run scripts/compute_norm_stats.py --config_name "your config name"
# 开启训练,model name会影响你保存后model的名称和你wandb任务的名称
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py "your config name" --exp-name="model name" --overwrite
```

### step4. 开启本地推理
为大家提供了一个简洁的模版, 当然还有一个我封装过的`pi_model.py`, 可以魔改一下.
``` python
from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# 你的config name, 定义了你输入输出的数据格式
config = _config.get_config("your config name") # pi0_base_aloha_lora/ pi0_base_aloha_full
checkpoint_dir = "your ckpt dir" #可以是你训练好的模型
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
example = {
    # 格式参考你的config里面的格式, 是一个dict, 索引对应名称
    "state": state, 
    "images": {
        "cam_high": img_front,
        "cam_left_wrist": img_left,
        "cam_right_wrist": img_right,
    },
    "prompt": self.instruction,
}

result = policy.infer(example)
```