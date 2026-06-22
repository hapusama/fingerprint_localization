import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse


def visualize_param_history(history_file, output_dir=None):
    """
    可视化参数冻结/解冻历史
    
    Args:
        history_file: 包含参数历史的JSON文件路径
        output_dir: 输出图像的目录，如果为None则仅显示不保存
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = [entry['epoch'] for entry in history]
    trainable = np.array([entry['trainable'] for entry in history])
    frozen = np.array([entry['frozen'] for entry in history])
    trainable_ratio = np.array([entry['trainable_ratio'] for entry in history])
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 子图1: 参数数量堆叠图
    ax1.stackplot(epochs, trainable, frozen, labels=['Trainable', 'Frozen'], 
                 colors=['#3498db', '#e74c3c'], alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_title('Parameter Counts Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Number of Parameters')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 为参数数量添加注释
    for i, e in enumerate(epochs):
        if i == 0 or i == len(epochs)-1 or (i > 0 and trainable[i] != trainable[i-1]):
            ax1.text(e, trainable[i] + frozen[i]/2, f"{int(trainable[i]):,}", 
                    ha='center', va='center', fontweight='bold')
    
    # 子图2: 可训练参数比例
    ax2.plot(epochs, trainable_ratio, 'o-', linewidth=2, markersize=8, color='#27ae60')
    
    # 添加解冻点注释
    for i in range(1, len(epochs)):
        if trainable_ratio[i] > trainable_ratio[i-1]:
            ax2.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5)
            ax2.text(epochs[i], trainable_ratio[i], f"{trainable_ratio[i]:.2%}",
                   ha='right', va='bottom')
    
    ax2.set_title('Trainable Parameter Ratio Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Trainable Parameter Ratio')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'param_history.png'), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(output_dir, 'param_history.png')}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize parameter freeze/unfreeze history')
    parser.add_argument('--history', type=str, required=True, help='Path to parameter history JSON file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save the output figure')
    
    args = parser.parse_args()
    visualize_param_history(args.history, args.output)
