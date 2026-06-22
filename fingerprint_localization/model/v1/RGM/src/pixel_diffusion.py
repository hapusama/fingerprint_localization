import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .denoising_diffusion_process import *
from .autoencoder import *

#是核心扩散模型，处理信号的正向扩散与反向去噪。
class PixelDiffusion(pl.LightningModule):

    def __init__(self,
                 train_dataset,
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-3, 
                 loss_fn=F.mse_loss,
                 schedule="cosine",
                 num_timesteps=1000,
                 sampler=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size=batch_size
        input_dim = 1
        #主要有前向传播的内容都在这里了
        self.model = DenoisingDiffusionConditionalProcess(input_dim, 
                                                          channels=2, 
                                                          dim_mults=(1, 2, 4, 8),
                                                          loss_fn=loss_fn, 
                                                          schedule=schedule, 
                                                          num_timesteps=num_timesteps, 
                                                          sampler=sampler)
    # todo generate debug一下这个forward函数，为什么生成数据全是1
    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.output_T(self.model(*args, **kwargs))
        # return self.model(*args, **kwargs)

    def input_T(self, input):
        # return input.clip(-1, 1)
        return input
    

    def output_T(self, input):
        # return input.clip(-1, 1)
        return input

    def training_step(self, batch_data, batch_idx):   
        signal_vec, location_vec, _ = batch_data
        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)
        # loss = self.model.p_loss(signal_vec, location_vec)
        self.log('train_loss', loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True,  # 在进度条右侧显示实时损失
                 logger=True)    # 同时记录到TensorBoard
        
        return loss
            
    def validation_step(self, batch_data, batch_idx):
        signal_vec, location_vec, _ = batch_data

        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)
        # loss = self.model.p_loss(signal_vec, location_vec)
        # 修改点2: 验证损失也显示在进度条
        self.log('val_loss', loss, 
                 on_epoch=True,   # 验证通常只关注epoch平均
                 prog_bar=True, 
                 logger=True)
        return loss
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None


    def configure_optimizers(self):
        return  torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)

#让模型学习 “位置→信号” 的映射，训练时利用位置向量作为条件，优化去噪损失
# ，最终实现根据位置生成对应信号（或恢复信号），服务于定位任务
class PixelDiffusionConditional_v2(PixelDiffusion):
    def __init__(self,
                 train_dataset, 
                 input_dim=4,
                 loc_dim=3,
                 channels=1, 
                 dim_mults=(1, 2, 4, 8),
                 valid_dataset=None, 
                 batch_size=1,
                 lr=1e-3, 
                 loss_fn=F.mse_loss,
                 schedule="cosine",
                 num_timesteps=1000,
                 signal_feature_dim=8,
                 sampler=None):
        pl.LightningModule.__init__(self)
        self.signal_feature_dim = signal_feature_dim
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size=batch_size
        self.lr = lr
        # 新增：初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # input_dim: [2,16]
        self.cha = channels
        self.dim=input_dim
        self.model = DenoisingDiffusionConditionalProcess(input_dim=input_dim, 
                                                          loc_dim=loc_dim,
                                                          channels=channels, 
                                                          dim_mults=dim_mults,
                                                          loss_fn=loss_fn, 
                                                          schedule=schedule, 
                                                          num_timesteps=num_timesteps, 
                                                          sampler=sampler,signal_feature_dim=signal_feature_dim)
    
    def training_step(self, batch_data, batch_idx):   
        signal_vec, location_vec, _ = batch_data
        # loss = self.model.p_loss(signal_vec, location_vec)
        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)

        self.log('train_loss', loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True,  # 在进度条右侧显示实时损失
                 logger=True)    # 同时记录到TensorBoard
        
        return loss
            
    def validation_step(self, batch_data, batch_idx):
        signal_vec, location_vec, _ = batch_data
        # loss = self.model.p_loss(signal_vec, location_vec)
        loss = self.model.p_loss(self.input_T(signal_vec), location_vec)
        # 修改点2: 验证损失也显示在进度条
        self.log('val_loss', loss, 
                 on_epoch=True,   # 验证通常只关注epoch平均
                 prog_bar=True, 
                 logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    def on_train_epoch_end(self):
        # 获取当前epoch的聚合损失值
        train_loss = self.trainer.callback_metrics['train_loss_epoch'].item()
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        
        # 保存到历史记录
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        # 实时打印历史
        print(f"\nEpoch {self.current_epoch:02d} Summary:")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print("History:")
        for i, (t_loss, v_loss) in enumerate(zip(self.history['train_loss'], self.history['val_loss'])):
            print(f"Epoch {i:02d}: {t_loss:.3f} / {v_loss:.3f}")

