import argparse
import yaml
import os
from argparse import Namespace

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args_freeze():
    """解析冻结参数配置"""
    # 构建配置文件路径
    config_path = r"configs\freeze_config.yml"
    
    # 读取 YAML 文件
    try:
        with open(config_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        
        # 将配置字典转换为命名空间对象
        args = Namespace(**config)
    except (FileNotFoundError, yaml.YAMLError):
        print(f"Warning: Could not load freeze config from {config_path}")
        args = Namespace(
            freeze_params={
                "enable_gradual_unfreeze": True,
                "initial_trainable_layers": ["class_emb", "final_conv"],
                "unfreeze_schedule": {
                    0.2: ["mid_block", "mid_attn"],
                    0.4: ["ups"],
                    0.6: ["downs"],
                    0.8: ["time_mlp"]
                }
            }
        )
    
    return args


def parse_args_location():
    # 构建配置文件路径
    config_path = r"configs\location_vector.yml"
    
    # 读取 YAML 文件
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 将配置字典转换为命名空间对象
    args = Namespace(**config)
    
    return args

def parse_args_pretrain():

    # 构建配置文件路径
    config_path = r"configs\pretrain.yml"
    
    # 读取 YAML 文件
    with open(config_path, 'r',encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # 将配置字典转换为命名空间对象
    args = Namespace(**config)
    
    return args

def parse_args_finetune():

    # 构建配置文件路径
    config_path = r"configs\finetune.yml"
    
    # 读取 YAML 文件
    with open(config_path, 'r',encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # 将配置字典转换为命名空间对象
    args = Namespace(**config)
    
    return args

def parse_args_vae():
    # 构建配置文件路径
    config_path = r"configs\vae.yml"
    print(f"Using configuration file: {config_path}\n")
    # 读取 YAML 文件
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)   

    # 将配置字典转换为命名空间对象
    args = Namespace(**config)

    return args
