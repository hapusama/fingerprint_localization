from src.Vae import *
import torch
from pytorch_lightning import loggers
import pytorch_lightning as pl
from src.parameter_paser import parse_args_vae
from src.dataset import ComplexDatasetLocs
from src.dataset import generate_three_dataset_v2
from torch.utils.data import DataLoader
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
if __name__ == '__main__':
    args = parse_args_vae()
    data_path=args.pretrain_dataset_pth
    num_epochs = args.num_epochs
    valid_frac = args.valid_frac
    test_frac = args.test_frac
    save_model_path = args.save_model_path
    
    loaded = torch.load(data_path)
    print("Using pretrain dataset: ", data_path)
    rssi=loaded['rssi']   
    snr=loaded['snr']
    label=loaded['label']

    location_vector_path = args.location_vector_path
    complex_dataset=ComplexDatasetLocs(rssi, 
                                       snr, 
                                       label, 
                                       location_vector_path
                                       )
    train_data_set, valid_data_set, test_data_set = generate_three_dataset_v2(complex_dataset,
                                                            valid_frac,
                                                            test_frac)

    input_dim = args.input_dim
    latent_dim = args.latent_dim
    condition_dim = args.condition_dim
    dim_mults = args.dim_mults
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    print(f"Model Configuration:")
    print(f"- Input Dimension: {input_dim}")
    print(f"- Condition Dimension: {condition_dim}")
    print(f"- Latent Dimension: {latent_dim}")
    print(f"- Dimension Multipliers: {dim_mults}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Learning Rate: {learning_rate}")

    vae_model = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        dim_mults=dim_mults,
        learning_rate=learning_rate)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model\\v1\\output\\vae\\lossmin",
        filename="vae_pretrain_lossmin",  # seems does not used
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=1,
        verbose=True)      
    train_loader=DataLoader(train_data_set,batch_size=batch_size, shuffle=True, num_workers=4,persistent_workers=True)
    val_loader=DataLoader(valid_data_set,batch_size=batch_size, shuffle=False, num_workers=4,persistent_workers=True)                              
    lr_monitor=pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    # 新增早停回调（监控 val_loss）
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss_epoch",    # 监控验证损失
        patience=25,           # 连续10个epoch未改善则停止
        mode="min",            # 监控指标越小越好
        verbose=True           # 打印停止信息
    )
    trainer=pl.Trainer(max_epochs=num_epochs,
                       accelerator='gpu',
                       devices=[0],
                       callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
                       enable_progress_bar=True,
                       log_every_n_steps=10)
    trainer.fit(vae_model,train_loader,val_loader)
    trainer.save_checkpoint(save_model_path)