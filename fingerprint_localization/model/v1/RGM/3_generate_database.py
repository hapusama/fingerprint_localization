import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from src.parameter_paser import parse_args_finetune,parse_args_pretrain
from src.dataset import ComplexDatasetLocs, ComplexDataset_real_imagary_v2
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from src.utils import get_features_by_label_v4
import time, json
import psutil
import threading
import statistics
from typing import List, Dict


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args_finetune()
    # args= parse_args_pretrain()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"

    num_locs = args.num_locs

    # using data from area b
    data_path_area_2 = data_path = os.path.join(input_dir, args.data_name_ft)
    print(f"data_path_area_2: {data_path_area_2}")
    loaded = torch.load(data_path_area_2)
    rssi = loaded['rssi']
    snr = loaded['snr']
    label = loaded['label']
    location_vector_path = os.path.join(output_dir, args.location_vector_name)

    complex_dataset = ComplexDatasetLocs(rssi, 
                                         snr, 
                                         label, 
                                         location_vector_path
                                         )
    
    input_dim = args.input_dim
    batch_si = args.batch_size_rgm
    learning_rate = args.learning_rate_rgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    dimension_scale = args.channel_dimension_scale
    signal_feature_dim=args.signal_feature_dim
    loaded_fine_tuned_rgm = os.path.join(output_dir, args.rgm_fine_tune_path)
    # loaded_fine_tuned_rgm=r"model\v1\output\2_finetuned_rgm.ckpt"
    # loaded_fine_tuned_rgm=r"model\v1\output\1_pretrained_rgm.ckpt"
    # loaded_fine_tuned_rgm = r"model\v1\output\lossmin\val_loss_finetune.ckpt"
    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)

    # prepare logs dir for metrics
    rgm_logs = os.path.join(output_dir, f"rgm_log")
    os.makedirs(rgm_logs, exist_ok=True)

    # --- Whole-run resource sampler ---
    class ResourceSampler(threading.Thread):
        """Background thread to sample CPU, RAM, and (optionally) CUDA memory during generation."""
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

    print("\nThe loaded diffusion model: {}\n".format(loaded_fine_tuned_rgm))
    # 这里先不用 loaded_fine_tuned_rgm
    diffusion_model = PixelDiffusionConditional_v2.load_from_checkpoint(checkpoint_path=loaded_fine_tuned_rgm, 
                                                                train_dataset=complex_dataset, 
                                                                input_dim=input_dim, 
                                                                loc_dim=loc_dim, 
                                                                channels=data_channels, 
                                                                dim_mults=dimension_scale, 
                                                                valid_dataset=complex_dataset, 
                                                                batch_size=batch_si, 
                                                                lr=learning_rate, 
                                                                loss_fn=model_loss, 
                                                                schedule=schedule, 
                                                                num_timesteps=num_timesteps, 
                                                                sampler=sampler_ddpm,signal_feature_dim=signal_feature_dim)
    
    diffusion_model.to(device)
    input_vec, _, _ = complex_dataset[0]
    #data_dimension, length = input_vec.shape
    data_dimension=data_channels
    length=input_dim
    # for saving generated and real collected data
    x_generated_list = []
    x_real_list = []
    loc_vec_list = []
    loc_int_list = []
    total_generated = 0
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
        # generate data for each location x全都代表csi
        for loc_int in range(num_locs):
            print("\nGenerating CSI data for Location ID: {}\n".format(loc_int))
            # real_data[0]: [4,]
            real_data, loc_tensor, loc_int_tensor = get_features_by_label_v4(complex_dataset, loc_int)
            if len(real_data) == 1:
                print("This point does not exit")
                continue
            batch_input = loc_tensor.to(device)
            # 600
            number_samples_generated = real_data.shape[0]

            if number_samples_generated > 600:    # 限制生成数量，提高生成过程速度
                number_samples_generated = 600
                batch_input = batch_input[:number_samples_generated]
                real_data = real_data[:number_samples_generated]
                loc_tensor = loc_tensor[:number_samples_generated]
                loc_int_tensor = loc_int_tensor[:number_samples_generated]

            data_shape = [number_samples_generated, 1, length]  # todo：把channel写入yml
            diffusion_model.eval()
            with torch.no_grad():   # batch_input: 位置向量
                generated_data = diffusion_model(data_shape, batch_input, sampler=sampler_ddpm, verbose=True)

            selected_feature = batch_input[:, -2:]
            selected_feature = selected_feature.unsqueeze(1)
            loc_tensor_last2 = loc_tensor[:, -2:]
            generated_data = torch.cat((generated_data, selected_feature), dim=2)
            real_data = torch.cat((real_data, loc_tensor_last2), dim=1)
            if getattr(args, "calibrate_generated_stats", False):
                real_signal = real_data[:, :length].to(generated_data.device)
                gen_signal = generated_data[:, :, :length]
                real_mean = real_signal.mean(dim=0, keepdim=True).unsqueeze(1)
                real_std = real_signal.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6).unsqueeze(1)
                gen_mean = gen_signal.mean(dim=0, keepdim=True)
                gen_std = gen_signal.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                gen_signal = (gen_signal - gen_mean) / gen_std
                gen_signal = gen_signal * real_std + real_mean
                generated_data[:, :, :length] = gen_signal.clamp(-6.0, 6.0)

            x_generated_list.append(generated_data.cpu())
            print("generated_data shape: ", generated_data.shape)
            print("generated_data : ", generated_data)
            print("real_data:", real_data)
            x_real_list.append(real_data.cpu())
            loc_vec_list.append(loc_tensor.cpu())
            loc_int_list.append(loc_int_tensor.cpu())
            total_generated += int(generated_data.size(0))

        tensor_generated_x = torch.cat(x_generated_list, dim=0)
        tensor_real_x = torch.cat(x_real_list, dim=0)
        tensor_loc_vec = torch.cat(loc_vec_list, dim=0)
        tensor_loc_int = torch.cat(loc_int_list, dim=0)

        complex_dataset_generated = ComplexDataset_real_imagary_v2(
            tensor_generated_x,
            tensor_real_x,
            tensor_loc_vec,
            tensor_loc_int
        )

        data_path_area_fake = data_path = os.path.join(output_dir, args.data_name_fake)
        torch.save(complex_dataset_generated, data_path_area_fake)
    finally:
        # Stop sampler and join
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
            model_class = type(diffusion_model).__name__
            total_params = sum(p.numel() for p in diffusion_model.parameters())
            trainable_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
            metrics['detected_model_class'] = model_class
            metrics['detected_model_total_params'] = total_params
            metrics['detected_model_trainable_params'] = trainable_params
        except Exception:
            pass

        # throughput / generated samples
        try:
            metrics['samples_generated'] = int(total_generated)
            metrics['samples_per_second'] = total_generated / total_seconds if total_seconds > 0 else None
        except Exception:
            pass

        # write JSON
        out_path = os.path.join(rgm_logs, 'generate_run_metrics.json')
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Wrote generation metrics to: {out_path}")
        except Exception as e:
            print("Failed to write generation metrics:", e)



