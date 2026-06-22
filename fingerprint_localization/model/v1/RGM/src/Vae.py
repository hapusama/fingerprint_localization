import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

class Residual(nn.Module):
    def __init__(self,fn):
        super(Residual, self).__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) w -> b h c w', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c w -> b (h c) w', h=self.heads, w=w)

        return self.to_out(out)

class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, *, norm=True):
        super().__init__()
        self.ds_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度可分离卷积
        
        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * 2, 1, padding=0),  # 升维
            nn.GELU(),
            nn.Conv1d(dim_out * 2, dim_out, 1, padding=0)  # 降维
        )
        self.res_conv = nn.Conv1d(dim, dim_out, 1, padding=0) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.conv = nn.Conv1d(dim, dim * 2, kernel_size=2, stride=2)
        self.conv = nn.Linear(dim, dim * 2)
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim // 2, kernel_size=1)
        # self.conv = nn.Linear(dim, dim // 2)

    def forward(self, x):
        return self.conv(x)


class ConditionalVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim=32,
        input_dim=6,
        condition_dim=8,
        dim_mults=(1, 2, 4, 8),  # U-Net式的维度倍数
        learning_rate=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.lr = learning_rate
        
        # 初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 条件编码器：将条件向量编码到与输入特征相同的维度
        self.condition_emb = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # 编码器
        self.encoder_in = nn.Conv1d(2, input_dim, 1)  # 2是拼接后的通道数
        
        # 构建U-Net式的编码器下采样路径
        dims = [input_dim, *map(lambda m: input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.encoder_downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.encoder_downs.append(nn.ModuleList([
                ConvBlock(dim_in, dim_out, norm=(ind != 0)),
                ConvBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),  # 添加注意力层
                DownSample(dim_out)
            ]))
            
        # 中间层
        mid_dim = dims[-1]
        self.encoder_mid = ConvBlock(mid_dim, mid_dim)
        self.encoder_mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        
        # 均值和方差预测层
        self.fc_mu = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mid_dim, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mid_dim, latent_dim)
        )
        
        # 解码器
        # 自定义展开层
        class Expander(nn.Module):
            def __init__(self, seq_len):
                super().__init__()
                self.seq_len = seq_len
            
            def forward(self, x):
                return x.unsqueeze(-1).expand(-1, -1, self.seq_len)

        # 先将潜变量和条件转换为特征图
        self.decoder_in = nn.Sequential(
            nn.Linear(latent_dim + input_dim, mid_dim * 2),
            nn.GELU(),
            nn.Linear(mid_dim * 2, mid_dim),
            Expander(input_dim)  # 使用自定义Module替代lambda函数
        )
        
        # 构建U-Net式的解码器上采样路径
        self.decoder_ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])): # (24,48) (12,24) (6,12)
            self.decoder_ups.append(nn.ModuleList([
                ConvBlock(dim_in * 2, dim_out),
                ConvBlock(dim_out, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),  # 添加注意力层
                UpSample(dim_out)
            ]))
            
        # 最终输出层
        self.final_conv = nn.Sequential(
            ConvBlock(dims[0], dims[0]),
            nn.Conv1d(dims[0], 1, 1),  # 输出单通道
            nn.Sigmoid()
        )

    def encode(self, x, c):
        # x: [batch_size, input_dim]
        # c: [batch_size, condition_dim]
        
        # 对条件进行编码，使其维度与输入特征匹配
        c = self.condition_emb(c)  # [batch_size, input_dim]
        
        # 将特征和条件重塑为序列形式并拼接
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        c = c.unsqueeze(1)  # [batch_size, 1, input_dim]
        x = torch.cat([x, c], dim=1)  # [batch_size, 2, input_dim]
        
        # 初始卷积
        x = self.encoder_in(x)
        
        # 保存中间特征用于跳跃连接
        h = []
        
        # 下采样路径
        for conv1, conv2, attn, downsample in self.encoder_downs:
            x = conv1(x)
            x = conv2(x)
            x = attn(x)  # 应用注意力
            h.append(x)
            x = downsample(x)
        
        # 中间处理
        x = self.encoder_mid(x)
        x = self.encoder_mid_attn(x)  # 应用中间层注意力
        
        # 预测均值和方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

    def decode(self, z, c):
        # z: [batch_size, latent_dim]
        # c: [batch_size, condition_dim]
        
        # 对条件进行编码
        c = self.condition_emb(c)  # [batch_size, input_dim]
        
        # 将潜变量和条件拼接
        z = torch.cat([z, c], dim=1)  # [batch_size, latent_dim + input_dim]
        
        # 转换为特征图
        x = self.decoder_in(z)  # [batch_size, mid_dim, input_dim]
        
        # 上采样路径
        for conv1, conv2, attn, upsample in self.decoder_ups:
            x = conv1(x)
            x = conv2(x)
            x = attn(x)  # 应用注意力
            x = upsample(x)
        
        # 最终输出
        x = self.final_conv(x)  # [batch_size, 1, input_dim]
        x = x.squeeze(1)  # [batch_size, input_dim]
        
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def on_train_epoch_end(self):
        # 获取当前epoch的聚合损失值
        train_loss = self.trainer.callback_metrics['train_loss_epoch'].item()
        val_loss = self.trainer.callback_metrics['val_loss_epoch'].item()
        
        # 保存到历史记录
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        # 实时打印历史
        print(f"\nEpoch {self.current_epoch:02d} Summary:")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print("History:")
        for i, (t_loss, v_loss) in enumerate(zip(self.history['train_loss'], self.history['val_loss'])):
            print(f"Epoch {i:02d}: {t_loss:.3f} / {v_loss:.3f}")

    def training_step(self, batch, batch_idx):
        x, c ,loc_id= batch
        recon_x, mu, logvar = self(x, c)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        loss=loss/batch[0].size(0)  # 平均损失
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, c, loc_id = batch
        recon_x, mu, logvar = self(x, c)    # 前向传播
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        loss=loss/batch[0].size(0)  # 平均损失
        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        #学习率下降策略
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True),
            'monitor': 'val_loss_epoch',  # 监控的指标名要和log一致
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
