import os
import torch
import yaml
from src.Vae import ConditionalVAE
from src.parameter_paser import parse_args_vae
from src.dataset import ComplexDatasetLocs, ComplexDataset_real_imagary_v2
from src.dataset import generate_three_dataset_v2

if __name__ == '__main__':
    # 读取配置文件
    config = parse_args_vae()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    data_path = config.dataset_ft
    print(f"Using dataset: {data_path}")
    loaded = torch.load(data_path)
    rssi = loaded['rssi']
    snr = loaded['snr']
    label = loaded['label']
    location_vector_path = config.location_vector_path
    complex_dataset = ComplexDatasetLocs(rssi, snr, label, location_vector_path)

    # 读取参数
    input_dim = config.input_dim
    condition_dim = config.condition_dim
    dim_mults = config.dim_mults
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    latent_dim = config.latent_dim
    save_model_path = config.save_model_path

    # 加载VAE模型
    vae = ConditionalVAE.load_from_checkpoint(
        checkpoint_path=save_model_path,
        input_dim=input_dim,
        condition_dim=condition_dim,
        dim_mults=dim_mults,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
    )
    vae.to(device)

    # 生成数据
    x_generated_list = []
    x_real_list = []
    loc_vec_list = []
    loc_int_list = []

    # 获取所有location id
    all_loc_ids = torch.unique(label)
    for loc_int in all_loc_ids:
        loc_int = loc_int.item() if hasattr(loc_int, 'item') else int(loc_int)
        print(f"\nGenerating data for Location ID: {loc_int}\n")
        # 获取该location的所有数据
        real_data, loc_tensor, loc_int_tensor = [], [], []
        for i in range(len(complex_dataset)):
            x, c, l_id = complex_dataset[i]
            if l_id == loc_int:
                real_data.append(x.unsqueeze(0))
                loc_tensor.append(c.unsqueeze(0))
                loc_int_tensor.append(torch.tensor([l_id]))
        if len(real_data) == 0:
            continue
        real_data = torch.cat(real_data, dim=0).to(device)
        loc_tensor = torch.cat(loc_tensor, dim=0).to(device)
        loc_int_tensor = torch.cat(loc_int_tensor, dim=0).to(device)
        number_samples_generated = real_data.shape[0]
        if number_samples_generated > 600:
            number_samples_generated = 600
            real_data = real_data[:600]
            loc_tensor = loc_tensor[:600]
            loc_int_tensor = loc_int_tensor[:600]
        # VAE生成
        with torch.no_grad():
            z = torch.randn(number_samples_generated, latent_dim).to(device)
            generated_data = vae.decode(z, loc_tensor)
        # 拼接location后两维
        selected_feature = loc_tensor[:, -2:].unsqueeze(1)  # (N,1,2)
        generated_data = torch.cat((generated_data.unsqueeze(1), selected_feature), dim=2)  # (N,1,input_dim+2)
        real_data = torch.cat((real_data, loc_tensor[:, -2:]), dim=1)  # (N,input_dim+2)
        x_generated_list.append(generated_data.cpu())
        x_real_list.append(real_data.cpu())
        loc_vec_list.append(loc_tensor.cpu())
        loc_int_list.append(loc_int_tensor.cpu())

    tensor_generated_x = torch.cat(x_generated_list, dim=0)
    tensor_real_x = torch.cat(x_real_list, dim=0)
    tensor_loc_vec = torch.cat(loc_vec_list, dim=0)
    tensor_loc_int = torch.cat(loc_int_list, dim=0)

    complex_dataset_generated = ComplexDataset_real_imagary_v2(
        tensor_generated_x, tensor_real_x, tensor_loc_vec, tensor_loc_int
    )

    data_path_area_fake = config.data_path_area_fake
    torch.save(complex_dataset_generated, data_path_area_fake)
    print(f"VAE生成数据已保存到: {data_path_area_fake}")
