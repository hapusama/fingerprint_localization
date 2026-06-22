import torch.nn as nn
from typing import List, Dict, Any
import pytorch_lightning as pl
import json
import os
from pathlib import Path


def freeze_model_except(model, unfrozen_names):
    """
    冻结模型除了指定名称之外的所有参数
    
    Args:
        model: 需要冻结参数的模型
        unfrozen_names: 不需要冻结的参数名称列表，支持部分匹配
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 显示主要模块列表
    print("\n=== 主要模块列表 ===")
    main_modules = set()
    for name, _ in model.named_children():
        main_modules.add(name)
    print(f"{', '.join(sorted(main_modules))}")
    
    # 解冻指定的参数
    unfrozen_modules = {}  # 用于记录每个模块解冻的参数数量
    matched_any = False
    
    for name, param in model.named_parameters():
        for unfrozen_name in unfrozen_names:
            if unfrozen_name in name:
                matched_any = True
                param.requires_grad = True
                
                # 记录解冻的模块和参数数量
                if unfrozen_name not in unfrozen_modules:
                    unfrozen_modules[unfrozen_name] = 0
                unfrozen_modules[unfrozen_name] += param.numel()
                break
    
    if not matched_any:
        print(f"\n⚠️ WARNING: None of the module names {unfrozen_names} matched any parameters!")
        print("This means NO parameters will be trainable! Check your configuration.")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n已解冻模块:")
    for module, params in unfrozen_modules.items():
        print(f"- {module}: {params:,} 参数")
    
    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"冻结参数: {total_params-trainable_params:,} ({1-trainable_params/total_params:.2%})")
    

def unfreeze_module(model, module_names):
    """
    解冻模型中指定模块的参数
    
    Args:
        model: 需要解冻参数的模型
        module_names: 需要解冻的模块名称列表，支持部分匹配
    """
    unfrozen_modules = {}  # 用于记录每个模块解冻了多少参数
    matched_any = False
    
    # 只在调试模式下打印所有主要模块
    print(f"\n=== 可用模块名 ===")
    main_modules = set()
    for name, _ in model.named_parameters():
        main_module = name.split('.')[0]
        if main_module not in main_modules:
            main_modules.add(main_module)
    print(f"{', '.join(sorted(main_modules))}")
    
    # 解冻匹配的模块
    for name, param in model.named_parameters():
        for module_name in module_names:
            if module_name in name:
                matched_any = True
                if not param.requires_grad:  # 只记录新解冻的参数
                    param.requires_grad = True
                    
                    # 提取主模块名称
                    main_module = module_name
                    if module_name not in unfrozen_modules:
                        unfrozen_modules[module_name] = 0
                    unfrozen_modules[module_name] += param.numel()
    
    if not matched_any:
        print(f"\n⚠️ WARNING: None of the module names {module_names} matched any parameters!")
        print("Check that your module names in freeze_config.yml match the actual parameter names in the model.")
    elif not unfrozen_modules:
        print(f"\n⚠️ WARNING: Modules {module_names} matched some parameters, but they were already unfrozen!")
    else:
        print(f"\n解冻模块: {', '.join(unfrozen_modules.keys())}")
        for module, params in unfrozen_modules.items():
            print(f"- {module}: {params:,} 参数")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}, 比例: {trainable_params/total_params:.2%}")


class GradualUnfreezeCallback(pl.Callback):
    """
    逐步解冻模型参数的回调函数
    """
    def __init__(self, unfreeze_schedule):
        """
        Args:
            unfreeze_schedule: 字典，键为epoch，值为要在该epoch解冻的模块名称列表
        """
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule
        # 添加历史记录
        self.param_history = []
    
    # def log_param_stats(self, pl_module):
    #     """记录当前模型参数的冻结/解冻状态"""
    #     # 获取正确的模型实例
    #     if hasattr(pl_module, 'model'):
    #         model = pl_module.model
    #         model_name = "pl_module.model"
    #     else:
    #         model = pl_module
    #         model_name = "pl_module"
            
    #     # 只在第一个epoch输出模型主要模块结构
    #     if pl_module.current_epoch == 0:
    #         print(f"\n=== Model Main Modules ===")
    #         for name, _ in model.named_children():
    #             print(f"- {name}")
        
    #     total_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     frozen_params = total_params - trainable_params
    #     trainable_ratio = trainable_params / total_params
        
    #     # 记录到模型历史
    #     stats = {
    #         'epoch': pl_module.current_epoch,
    #         'total': total_params,
    #         'trainable': trainable_params,
    #         'frozen': frozen_params,
    #         'trainable_ratio': trainable_ratio
    #     }
    #     self.param_history.append(stats)
        
    #     # 记录到TensorBoard
    #     pl_module.log('trainable_params', trainable_params, on_step=False, on_epoch=True)
    #     pl_module.log('frozen_params', frozen_params, on_step=False, on_epoch=True)
    #     pl_module.log('trainable_ratio', trainable_ratio, on_step=False, on_epoch=True)
        
    #     # 打印当前状态
    #     print(f"\n=== Parameter Status - Epoch {pl_module.current_epoch} ===")
    #     print(f"Total parameters:     {total_params:,}")
    #     print(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2%})")
    #     print(f"Frozen parameters:    {frozen_params:,} ({1-trainable_ratio:.2%})")
        
    #     # 统计主要模块的可训练参数
    #     main_module_stats = {}
        
    #     # 预定义感兴趣的主要模块名称
    #     main_modules = ['model', 'time_mlp', 'class_emb', 'downs', 'ups', 'mid_block', 'mid_attn', 'encoder_downs', 
    #                     'encoder_mid', 'decoder_ups', 'final_conv']
        
    #     # 初始化主要模块统计
    #     for module in main_modules:
    #         main_module_stats[module] = {'total': 0, 'trainable': 0, 'status': 'N/A'}
        
    #     # 填充统计信息
    #     for name, param in model.named_parameters():
    #         # 提取主模块名称（通常是第一级路径）
    #         main_module = name.split('.')[0]
            
    #         # 对于特殊情况，检查更深层次的模块
    #         for module in main_modules:
    #             if module in name:
    #                 if module not in main_module_stats:
    #                     main_module_stats[module] = {'total': 0, 'trainable': 0, 'status': 'N/A'}
                    
    #                 main_module_stats[module]['total'] += param.numel()
    #                 if param.requires_grad:
    #                     main_module_stats[module]['trainable'] += param.numel()
        
    #     # 计算状态
    #     for module, stats in main_module_stats.items():
    #         if stats['total'] > 0:
    #             ratio = stats['trainable'] / stats['total']
    #             if ratio == 1.0:
    #                 stats['status'] = '完全解冻'
    #             elif ratio == 0.0:
    #                 stats['status'] = '完全冻结'
    #             else:
    #                 stats['status'] = f'部分解冻 ({ratio:.1%})'
        
    #     # 打印主要模块的训练状态
    #     print("\n=== Main Module Training Status ===")
    #     print(f"{'Module':<20} {'Status':<15} {'Trainable/Total'}")
    #     print("-" * 60)
        
    #     # 按训练参数比例排序（从高到低）
    #     for module, stats in sorted(main_module_stats.items(), 
    #                                key=lambda x: -x[1]['trainable']/max(x[1]['total'], 1) 
    #                                if x[1]['total'] > 0 else -1):
    #         if stats['total'] > 0:  # 只显示存在的模块
    #             print(f"{module:<20} {stats['status']:<15} {stats['trainable']:,}/{stats['total']:,}")
    def log_param_stats(self, pl_module):
        """记录当前模型参数的冻结/解冻状态"""
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module
        
        # 统计可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_ratio = trainable_params / total_params
        
        # 记录到历史
        stats = {
            'epoch': pl_module.current_epoch,
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_ratio': trainable_ratio
        }
        self.param_history.append(stats)
        
        # 打印当前状态
        print(f"\n=== 参数状态 - 第 {pl_module.current_epoch} 轮 ===")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_ratio:.2%})")
        print(f"冻结参数: {frozen_params:,} ({1-trainable_ratio:.2%})")
        
        # 统计主要模块的可训练参数
        main_modules = {
            'time_mlp': 0, 'class_emb': 0,
            'downs': 0, 'mid_block': 0, 'mid_attn': 0, 'ups': 0, 'final_conv': 0
        }
        main_modules_trainable = {k: 0 for k in main_modules.keys()}
        
        for name, param in model.named_parameters():
            for module in main_modules.keys():
                if module in name:
                    main_modules[module] += param.numel()
                    if param.requires_grad:
                        main_modules_trainable[module] += param.numel()
        
        # 打印主要模块的训练状态
        print("\n=== 主要模块训练状态 ===")
        print(f"{'模块':<15} {'状态':<15} {'可训练/总数'}")
        print("-" * 50)
        
        for module in main_modules.keys():
            total = main_modules[module]
            trainable = main_modules_trainable[module]
            if total > 0:
                ratio = trainable / total
                status = "完全解冻" if ratio == 1.0 else ("完全冻结" if ratio == 0.0 else f"部分解冻 ({ratio:.1%})")
                print(f"{module:<15} {status:<15} {trainable:,}/{total:,}")
                
    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        
        # 打印解冻计划（仅在第一个epoch）
        if current_epoch == 0:
            print("\n=== 解冻计划 ===")
            for epoch, modules in sorted(self.unfreeze_schedule.items()):
                print(f"Epoch {epoch}: 将解冻 {', '.join(modules)}")
        
        # 检查是否需要在当前epoch解冻某些层
        if current_epoch in self.unfreeze_schedule:
            modules_to_unfreeze = self.unfreeze_schedule[current_epoch]
            print(f"\n=== Epoch {current_epoch}: 解冻模块 {', '.join(modules_to_unfreeze)} ===")
            
            # 获取正确的模型实例
            if hasattr(pl_module, 'model'):
                target_model = pl_module.model
            else:
                target_model = pl_module
                
            unfreeze_module(target_model, modules_to_unfreeze)
        
        # 每个epoch开始时记录参数状态
        self.log_param_stats(pl_module)
   
    
    def on_fit_end(self, trainer, pl_module):
        """训练结束时，保存参数历史到文件"""
        # 创建输出目录
        output_dir = Path(trainer.logger.log_dir) / "param_history"
        output_dir.mkdir(exist_ok=True)
        
        # 将参数历史保存为JSON文件
        history_file = output_dir / "param_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.param_history, f, indent=2)
        
        print(f"\nParameter history saved to {history_file}")
        print("You can visualize the history using: python src/visualize_params.py --history " + str(history_file))
