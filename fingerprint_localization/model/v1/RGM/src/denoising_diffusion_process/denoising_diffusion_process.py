import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.simple_unet import *


class DenoisingDiffusionConditionalProcess(nn.Module):
    
    def __init__(self,
                 input_dim=4,
                 loc_dim=3,
                 channels=2,
                 dim_mults=(1, 2, 4, 8),
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None,signal_feature_dim=8):
        super().__init__()
        # self.trans_dim = signal_feature_dim #并非输入维度
        self.loc_dim = loc_dim
        self.channels = channels    #并非输入信道数
        self.loss_fn = loss_fn
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        # Forward Process
        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps, 
                                                      schedule=self.schedule)
        
        # Reverse Process. defaults to a DDPM sampler if None is provided
        self.sampler = DDPM_Sampler(num_timesteps=self.num_timesteps, 
                                    schedule=self.schedule) if sampler is None else sampler
        
        # noise prediction networks
        self.model = UnetComplexBlock(input_dim, 
                                      loc_dim=loc_dim,
                                      channels=channels, 
                                      dim_mults=dim_mults,signal_feature_dim=signal_feature_dim)

    # @torch.no_grad()
    # def forward(self,
    #             data_shape,
    #             condition,
    #             sampler=None,
    #             verbose=False):
    #     """
    #         forward() function triggers a complete inference cycle

    #         A custom sampler can be provided as an argument!
    #     """

    #     # read dimensions
    #     # b, c, h, w = x.shape
    #     b, h, w = data_shape

    #     device = next(self.model.parameters()).device
    #     condition = condition.to(device)
        
    #     # select sampler
    #     if sampler is None:
    #         sampler = self.sampler
            
    #     else:
    #         sampler.to(device)

    #     # time steps list
    #     num_timesteps = sampler.num_timesteps 
    #     it = reversed(range(0, num_timesteps))
        
    #     # x_t = torch.randn([b, c, h, w], device=device)
    #     x_t = torch.randn([b, h, w], device=device)

    #     for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:

    #         t = torch.full((b,), i, device=device, dtype=torch.long)
    #         z_t = self.model(x_t, t, condition)   # prediction of noise

    #         # call forward function of DDPM_Sampler Class: 
    #         # Given approximation of noise z_t in x_t predict x_(t-1)
    #         # prediction of next state
    #         x_t = sampler(x_t, t, z_t)
            
    #     print(x_t.shape)
    #     return x_t
    @torch.no_grad()
    def forward(self,
                data_shape,
                condition,
                sampler=None,
                verbose=False):
        """
            forward() function triggers a complete inference cycle

            A custom sampler can be provided as an argument!
        """

        # read dimensions
        # b, c, h, w = x.shape
        b, h, w = data_shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        
        # select sampler
        if sampler is None:
            sampler = self.sampler
            
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps 
        it = reversed(range(0, num_timesteps))
        
        # x_t = torch.randn([b, c, h, w], device=device)
        x_t = torch.randn([b, h, w], device=device)

        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:

            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t = self.model(x_t, t, condition)   # prediction of noise

            # call forward function of DDPM_Sampler Class: 
            # Given approximation of noise z_t in x_t predict x_(t-1)
            # prediction of next state
            x_t = sampler(x_t, t, z_t)
            
        print(x_t.shape)
        return x_t

    # x is signal condition is location vector
    def p_loss(self, x, condition):
        """
            Assumes output and input are in [-1,+1] range
        """        
        # batch size，channel(phase amplitude) input_dim
        # 当h与w不存在时，x.shape = (b, input_dim)
        if len(x.shape) == 2:
            b, h, w = x.shape[0], 1, x.shape[1]
            x = x.view(b, h, w)
        b,h,w = x.shape
        device = x.device
        #随机生成numsteps以内 256个不同整数
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()

        # call forward function of GaussianForwardProcess Class:  
        # Get noisy sample at t given x_0
        output_noisy, noise = self.forward_process(x, t, return_noise=True)

        noise_hat = self.model(output_noisy, t, condition)

        # apply loss
        return self.loss_fn(noise, noise_hat)
    

    