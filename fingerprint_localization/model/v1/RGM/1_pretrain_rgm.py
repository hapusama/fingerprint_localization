import os
import torch
from pytorch_lightning import loggers
import pytorch_lightning as pl
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.parameter_paser import parse_args_pretrain
from src.dataset import generate_three_dataset_v2, ComplexDatasetLocs
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from sklearn.preprocessing import MinMaxScaler
from src import EMA
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import time, json
import psutil
import threading
import statistics
from typing import List, Dict
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    args = parse_args_pretrain()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"

    frac_for_valid = args.frac_for_valid
    frac_for_test = args.frac_for_test
    # FLOOR3.pth
    data_path_area_1 = data_path = os.path.join(input_dir, args.data_name)
    loaded = torch.load(data_path_area_1)
    print(f"data_path_area_1: {data_path_area_1}")
    rssi = loaded['rssi']   # shape [24576,7]
    snr = loaded['snr']
    label = loaded['label']
    
    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    # 生成一个数据集, 32000个数据，每个数据有rssi, snr, label, location_vector
    complex_dataset = ComplexDatasetLocs(rssi, 
                                         snr, 
                                         label, 
                                         location_vector_path
                                         )
    
    train_data_set, valid_data_set, test_data_set = generate_three_dataset_v2(complex_dataset, 
                                                            frac_for_valid,
                                                            frac_for_test)

    input_dim = args.input_dim
    num_epochs = args.num_epochs_rgm
    batch_si = args.batch_size_rgm
    learning_rate = args.learning_rate_rgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    # 通道的倍数 1 2 4 8
    dimension_scale = args.channel_dimension_scale
    model_path_train_rgm = os.path.join(output_dir, args.rgm_pretrain_path)
    signal_feature_dim=args.signal_feature_dim
    rgm_logs = os.path.join(output_dir, f"rgm_log")
    os.makedirs(rgm_logs, exist_ok=True)

    # --- Whole-run resource sampler ---
    class ResourceSampler(threading.Thread):
        """Background thread to sample CPU, RAM, and (optionally) CUDA memory during training."""
        def __init__(self, sample_interval: float = 5.0):
            super().__init__(daemon=True)
            self.interval = sample_interval
            self._stop_evt = threading.Event()
            self.samples: List[Dict] = []
            self.proc = psutil.Process()

        def run(self):
            # Prime cpu_percent measurement
            psutil.cpu_percent(interval=None)
            while not self._stop_evt.is_set():
                ts = time.time()
                cpu = psutil.cpu_percent(interval=None)
                mem = self.proc.memory_info()
                vm = psutil.virtual_memory()
                sample = {
                    'timestamp': ts,
                    'cpu_percent': cpu,
                    'process_rss_bytes': mem.rss,
                    'process_vms_bytes': mem.vms,
                    'system_available_mem_bytes': vm.available,
                    'system_percent': vm.percent
                }
                # CUDA stats (if available)
                try:
                    if torch.cuda.is_available():
                        sample['cuda_current_allocated_bytes'] = torch.cuda.memory_allocated()
                        sample['cuda_current_reserved_bytes'] = torch.cuda.memory_reserved()
                except Exception:
                    pass

                self.samples.append(sample)
                # wait for interval or stop
                self._stop_evt.wait(self.interval)

        def stop(self):
            self._stop_evt.set()

        def get_summary(self):
            if not self.samples:
                return {}
            cpu_vals = [s['cpu_percent'] for s in self.samples if 'cpu_percent' in s]
            rss_vals = [s['process_rss_bytes'] for s in self.samples if 'process_rss_bytes' in s]
            vms_vals = [s['process_vms_bytes'] for s in self.samples if 'process_vms_bytes' in s]
            cuda_alloc = [s.get('cuda_current_allocated_bytes', 0) for s in self.samples]
            cuda_resv = [s.get('cuda_current_reserved_bytes', 0) for s in self.samples]
            return {
                'cpu_percent_avg': statistics.mean(cpu_vals) if cpu_vals else None,
                'cpu_percent_max': max(cpu_vals) if cpu_vals else None,
                'process_rss_max_bytes': max(rss_vals) if rss_vals else None,
                'process_vms_max_bytes': max(vms_vals) if vms_vals else None,
                'cuda_current_allocated_max_bytes': max(cuda_alloc) if any(cuda_alloc) else None,
                'cuda_current_reserved_max_bytes': max(cuda_resv) if any(cuda_resv) else None,
                'samples_count': len(self.samples)
            }

    sampler = ResourceSampler(sample_interval=5.0)

    tb_logger = loggers.TensorBoardLogger(save_dir=rgm_logs, 
                                        name='', 
                                        version="rgm_pretrain")
    # 降噪的过程（反向），加噪那一堆参数需要进一步了解，ddpm应该是降噪用的 下面那个是加噪的前向传播
    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model\\v1\\output\\lossmin",
        filename="val_loss_pretrain",  # seems does not used
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True)

    model = PixelDiffusionConditional_v2(train_data_set, 
                                        input_dim=input_dim, 
                                        loc_dim=loc_dim,
                                        channels=data_channels, 
                                        dim_mults=dimension_scale, 
                                        valid_dataset=valid_data_set, 
                                        batch_size=batch_si, 
                                        lr=learning_rate, 
                                        loss_fn=model_loss, 
                                        schedule=schedule, 
                                        num_timesteps=num_timesteps, 
                                        sampler=sampler_ddpm,signal_feature_dim=signal_feature_dim)

    train_loader = DataLoader(train_data_set, batch_size=batch_si, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(valid_data_set, batch_size=batch_si, shuffle=False, num_workers=4, persistent_workers=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    # 新增早停回调（监控 val_loss）
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",    # 监控验证损失
        patience=30,           # 连续30个epoch未改善则停止
        mode="min",            # 监控指标越小越好
        verbose=True           # 打印停止信息
    )

    trainer = pl.Trainer(max_epochs=num_epochs, 
                        callbacks=[EMA(0.9999),lr_monitor, checkpoint_callback,early_stop_callback], 
                        accelerator='gpu', 
                        devices=[0], 
                         enable_progress_bar=True,
                        check_val_every_n_epoch=1,
                        logger=tb_logger)
    # Reset CUDA peak stats if CUDA is available
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # Start background sampling
    sampler.start()

    overall_start = time.time()
    try:
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(model_path_train_rgm)
    except Exception as e:
        print("Training raised exception:", e)
        raise
    finally:
        # Stop sampler and wait for it
        sampler.stop()
        sampler.join(timeout=5.0)

        overall_end = time.time()
        total_seconds = overall_end - overall_start

        # Build metrics summary
        metrics = {}
        metrics['start_time'] = overall_start
        metrics['start_time_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(overall_start))
        metrics['end_time'] = overall_end
        metrics['end_time_iso'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(overall_end))
        metrics['duration_seconds'] = total_seconds

        # sampler summary
        sampler_summary = sampler.get_summary()
        metrics.update(sampler_summary)

        # Torch CUDA peak stats
        try:
            if torch.cuda.is_available():
                metrics['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                metrics['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                metrics['cuda_current_allocated_bytes'] = torch.cuda.memory_allocated()
                metrics['cuda_current_reserved_bytes'] = torch.cuda.memory_reserved()
        except Exception:
            pass

        # model params
        try:
            model_class = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metrics['detected_model_class'] = model_class
            metrics['detected_model_total_params'] = total_params
            metrics['detected_model_trainable_params'] = trainable_params
        except Exception:
            pass

        # throughput estimate
        try:
            steps_per_epoch = len(train_loader)
            # epochs run: trainer.current_epoch is 0-based and points to last completed epoch after fit
            epochs_run = getattr(trainer, 'current_epoch', num_epochs)
            # trainer.current_epoch may be last completed index, so epochs_run = current_epoch + 1 if training ran
            if hasattr(trainer, 'current_epoch'):
                epochs_run = trainer.current_epoch + 1
            samples_processed = int(steps_per_epoch * epochs_run * batch_si)
            metrics['samples_processed'] = samples_processed
            metrics['samples_per_second'] = samples_processed / total_seconds if total_seconds > 0 else None
        except Exception:
            pass

        # write JSON
        out_path = os.path.join(rgm_logs, 'pretrain_run_metrics.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Wrote training metrics to: {out_path}")
        except Exception as e:
            print("Failed to write training metrics:", e)

    if os.environ.get("RGM_WAIT_FOR_ENTER", "0") == "1":
        input("Training complete. Press Enter to exit...")
