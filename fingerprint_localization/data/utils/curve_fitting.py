import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义路径损耗模型公式: RSSI = A - 10 * n * log10(d)
def path_loss_model(d, A, n):
    return A - 10 * n * np.log10(d)

fitting_data_path=r'data\processedData\FLOOR3\all_data.csv'
location_vector_path=r'model\v1\output\location_vector_20m.csv'
# label_coordinate_path=r'PicdataProcessing\image.png'

# 空间采样间隔为9.7m
# area1_list=[0,1,2,3,4,5]
# area2_list=[6,7,8,9,10,11,12,13,14]
# area3_list=[15,16,17,18,19,20]

# 空间采样间隔为6.2m
# area1_list=[0,1,2,3,4,5,6,7,8,9,10]
# area2_list=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
# area3_list=[28,29,30,31,32,33,34,35,36]

# 空间采样间隔为14.9m
area1_list=[0,1,2,3,4]
area2_list=[5,6,7,8,9]
area3_list=[10,11,12,13]
PLM_save_path=r"model\v1\output\PLM_FLOOR3.csv"

def fit_path_loss_per_sf(data_df, df, area_lists, area_names, colors, path_loss_model):
    """
    为每个扩频因子(SF)和区域拟合路径损耗模型，并绘制结果。

    参数:
    data_df (pd.DataFrame): 包含RSSI、SF和location_id的DataFrame。
    df (pd.DataFrame): 包含location_id和真实距离的DataFrame。
    area_lists (list): 包含每个区域location_id列表的列表。
    area_names (list): 每个区域的名称列表。
    colors (list): 用于绘图的颜色列表。
    path_loss_model (function): 要拟合的路径损耗模型函数。
    """
    # 获取数据中所有唯一的扩频因子并排序
    sf_list = sorted(data_df['sf'].unique())
    
    # 1. 初始化一个列表来收集所有的参数数据
    all_params_data = []

    # 遍历每个扩频因子
    for sf in sf_list:
        # 为每个SF创建一个新的图形
        plt.figure(figsize=(8,6))
        print(f"\n=== SF={sf} ===")
        
        # 遍历每个定义的区域
        for area_idx, area_location_ids in enumerate(area_lists):
                
            all_distances = []
            all_rssis = []
            
            # 遍历当前区域中的每个位置ID
            for i in area_location_ids:
                # 筛选出当前位置ID和SF的数据
                sample_df = data_df[(data_df['location_id'] == i) & (data_df['sf'] == sf)]
                if sample_df.empty:
                    continue
                
                # 获取该位置的真实距离
                distance = df[df['location_id'] == i]['distance_true'].values[0]
                # 获取对应的RSSI值
                rssi = sample_df['realtime_average_rssi'].values 
                
                # 将距离和RSSI值添加到列表中
                all_distances.extend([distance] * len(rssi))
                all_rssis.extend(rssi)
            
            # 如果当前区域没有数据，则跳过
            if len(all_distances) == 0 or len(all_rssis) == 0:
                continue
            
            # 将列表转换为numpy数组以便计算
            all_distances = np.array(all_distances)
            all_rssis = np.array(all_rssis)
            
            # 使用curve_fit进行曲线拟合，p0是参数的初始猜测值
            popt, _ = curve_fit(path_loss_model, all_distances, all_rssis, p0=[-40, 2])
            
            # 生成用于绘制拟合曲线的距离数据点
            d_fit = np.linspace(all_distances.min(), all_distances.max(), 100)
            # 计算拟合曲线上的RSSI值
            rssi_fit = path_loss_model(d_fit, *popt)
            
            # 绘制原始数据点的散点图
            plt.scatter(all_distances, all_rssis, color=colors[area_idx], alpha=0.3, label=f'{area_names[area_idx]} Data')
            # 绘制拟合曲线
            plt.plot(d_fit, rssi_fit, color=colors[area_idx], label=f'{area_names[area_idx]} Fit: A={popt[0]:.2f}, n={popt[1]:.2f}')
            
            # 打印拟合得到的参数A和n
            print(f'{area_names[area_idx]} 拟合参数: A={popt[0]:.2f}, n={popt[1]:.2f}')
            # 准备要保存的数据
            param_data = {
                'sf': sf,
                'Area': area_names[area_idx],
                'A': popt[0],
                'n': popt[1]
            }
            # 2. 将当前参数字典添加到列表中
            all_params_data.append(param_data)

        # # 对area1重新进行拟合
        # idx_base = 6 # 计算网关到idx的path loss
        # average_rssi_idx = data_df[(data_df['location_id'] == idx_to_location_id[idx_base]) & (data_df['sf'] == sf)]['realtime_average_rssi'].mean()
        # # 计算area1中的点与idx为7的点之间的距离，并根据距离和rssi进行重新拟合area1
        # area1_distances = []
        # area1_rssis = []
        # # 重新便利area1
        # for i in area1_location_id_list:
        #     if i == idx_to_location_id[idx_base]:
        #         continue
        #     # 计算与idx_base的距离
        #     coords_i = (df[df['location_id'] == i]['true_x'].values[0],df[df['location_id'] == i]['true_y'].values[0])
        #     coords_base = (df[df['location_id'] == idx_to_location_id[idx_base]]['true_x'].values[0],
        #                    df[df['location_id'] == idx_to_location_id[idx_base]]['true_y'].values[0])
        #     distance_base = np.linalg.norm(np.array(coords_i) - np.array(coords_base))
        #     # 获取idx_base的真实距离
        #     distance_idx_base = df[df['location_id'] == idx_to_location_id[idx_base]]['distance_true'].values[0]
        #     distance = abs(distance_base - distance_idx_base)
        #     # 获取对应的RSSI值
        #     rssi_values = data_df[(data_df['location_id'] == i) & (data_df['sf'] == sf)]['realtime_average_rssi'].values
        #     # 创建与rssi_values相同长度的average_rssi_idx数组
        #     # average_rssi_array = np.full_like(rssi_values, average_rssi_idx)
        #     # rssi = rssi_values - average_rssi_array
        #     rssi = rssi_values
        #     if data_df[(data_df['location_id'] == i) & (data_df['sf'] == sf)]['realtime_average_rssi'].empty:
        #         continue
        #     area1_distances.extend([distance] * len(rssi))
        #     area1_rssis.extend(rssi)

        # if area1_distances and area1_rssis:
        #     area1_distances = np.array(area1_distances)
        #     area1_rssis = np.array(area1_rssis)
        #     popt_area1, _ = curve_fit(path_loss_model, area1_distances, area1_rssis, p0=[-40, 2])
        #     d_fit_area1 = np.linspace(area1_distances.min(), area1_distances.max(), 100)
        #     rssi_fit_area1 = path_loss_model(d_fit_area1, *popt_area1)
        #     plt.plot(d_fit_area1, rssi_fit_area1, color='m', linestyle='--', label=f'Area1 Refit: A={popt_area1[0]:.2f}, n={popt_area1[1]:.2f}')
        #     print(f'Area1 重新拟合参数: A={popt_area1[0]:.2f}, n={popt_area1[1]:.2f}')
        #     # 用重新拟合的参数覆盖当前sf下area1的数据
        #     for i, param in enumerate(all_params_data):
        #         if param['sf'] == sf and str(param['Area']) == '1':
        #             all_params_data[i] = {
        #                 'sf': sf,
        #                 'Area': '1',
        #                 'A': popt_area1[0],
        #                 'n': popt_area1[1]
        #             }
        #             break
        #     else:
        #         # 如果没有找到，则添加
        #         all_params_data.append({
        #             'sf': sf,
        #             'Area': '1',
        #             'A': popt_area1[0],
        #             'n': popt_area1[1]
        #         })
            
        
    # 3. 在所有循环结束后，将收集到的数据一次性写入CSV文件
    if all_params_data:
        pd.DataFrame(all_params_data).to_csv(PLM_save_path, index=False)

if __name__=="__main__":
    df = pd.read_csv(location_vector_path)
    idx_to_location_id = dict(zip(df['idx'], df['location_id']))
    area1_location_id_list = [idx_to_location_id[i] for i in area1_list]
    area2_location_id_list = [idx_to_location_id[i] for i in area2_list]
    area3_location_id_list = [idx_to_location_id[i] for i in area3_list]
    data_df = pd.read_csv(fitting_data_path)

    area_lists = [area1_location_id_list, area2_location_id_list, area3_location_id_list]
    area_names = [1, 2, 3]
    colors = ['r', 'g', 'b']

    fit_path_loss_per_sf(data_df, df, area_lists, area_names, colors, path_loss_model)
