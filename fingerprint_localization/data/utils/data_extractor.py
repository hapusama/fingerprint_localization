import pandas as pd
import os
import numpy as np
import torch
# 定义固定列名
fixed_columns = ['hdok', 'plok', 'none', 'nums', 'totalNums', 'average_rssi', 'snr', 'sf', 'tp', 'serial_size']
save_file_path= os.path.join(os.getcwd(), 'data', 'processedData')

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
def apply_kalman_filter(row, rssi_columns):
    rssi_values = row[rssi_columns].astype(float).values  # 转换为浮点数
    if len(rssi_values) > 0:  # 确保有数据可以进行滤波
        n_iter = len(rssi_values)
        sz = (n_iter,)
        Q = 1e-5  # process variance

        xhat = np.zeros(sz)
        P = np.zeros(sz)
        xhatminus = np.zeros(sz)
        Pminus = np.zeros(sz)
        K = np.zeros(sz)

        R = 0.1 ** 2  # estimate of measurement variance

        xhat[0] = rssi_values[0]
        P[0] = 1.0

        for k in range(1, n_iter):
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q

            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (rssi_values[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        row[rssi_columns[:len(xhat)]] = xhat
    return row

#将txt文件全都转化成csv文件
def txt_to_csv(file_path):
    current_data_path = os.path.join(os.getcwd(), 'data', 'rawData')
    file_path_copy=file_path
    file_path = os.path.join(current_data_path, file_path)
    files = os.listdir(file_path)
    
    # files: /data/rawData/{file_path}  
    for folder in files:
        # folder_path: ..../files/SF_
        folder_path = os.path.join(file_path, folder)
        if os.path.isdir(folder_path):
            txt_files = os.listdir(folder_path)
            for txt_file in txt_files:
                if txt_file.endswith('.txt'):
                    with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    data = []
               
                    for line in lines:
                        if line.startswith('HDOK') or line.startswith('PLOK'):
                            line = line.strip().split()
                            data.append(line)
                    if data:
                        serial_size = int(data[0][9])
                        for i in range(len(data)):
                            if len(data[i]) > len(fixed_columns) + serial_size:
                                data[i] = data[i][:len(fixed_columns) + serial_size]
                        columns = fixed_columns + [f'rssi_{i}' for i in range(serial_size)]
                        columns.append('location_id')
                        for i in range(len(data)):
                            data[i].append(txt_file.replace('.txt', '').replace('.0', ''))
                        df = pd.DataFrame(data, columns=columns)
                        rssi_columns = [col for col in df.columns if col.startswith('rssi_') and col != 'rssi_variance' and col != 'rssi_skewness' and col != 'rssi_kurtosis']
                        df = df.apply(lambda row: apply_kalman_filter(row, rssi_columns), axis=1)
                        augmented_data = []
                        for _, row in df.iterrows():
                            rssi_values = row[rssi_columns].astype(float).values
                            for _ in range(60):
                                noise = np.random.normal(0, 1, size=rssi_values.shape)
                                augmented_rssi = rssi_values + noise
                                augmented_row = row.copy()
                                augmented_row[rssi_columns] = augmented_rssi
                                augmented_data.append(augmented_row)
                        augmented_df = pd.DataFrame(augmented_data, columns=df.columns)
                        df = pd.concat([df, augmented_df], ignore_index=True)
                        df[rssi_columns] = df[rssi_columns].dropna().apply(pd.to_numeric, errors='coerce')
                        df['realtime_average_rssi'] = df[rssi_columns].mean(axis=1)
                        columns.append('realtime_average_rssi')
                        df['median_rssi'] = df[rssi_columns].median(axis=1)
                        columns.append('median_rssi')
                        df['mode_rssi'] = df[rssi_columns].mode(axis=1)[0]
                        columns.append('mode_rssi')
                        def calculate_variance_without_outliers(row):
                            rssi_values = row[rssi_columns].astype(float).values
                            return np.var(rssi_values)
                        df['rssi_variance'] = df.apply(calculate_variance_without_outliers, axis=1)
                        columns.append('rssi_variance')
                        def calculate_skewness(row):
                            rssi_values = row[rssi_columns].astype(float).values
                            return pd.Series(rssi_values).skew()
                        df['rssi_skewness'] = df.apply(calculate_skewness, axis=1)
                        def calculate_kurtosis(row):
                            rssi_values = row[rssi_columns].astype(float).values
                            return pd.Series(rssi_values).kurtosis()
                        df['rssi_kurtosis'] = df.apply(calculate_kurtosis, axis=1)
                        save_folder_path = os.path.join(save_file_path, file_path_copy)
                        save_folder_path=os.path.join(save_folder_path, folder)
                        if not os.path.exists(save_folder_path):
                            os.makedirs(save_folder_path)
                        df.to_csv(os.path.join(save_folder_path, txt_file.replace('.txt', '.csv')), index=False)
                        print(f"文件 {txt_file} 转换成功")
#将每一层的CSV文件合并成一个CSV文件为 all_data.csv                        
def csv_to_csv(file_floor,PLM_params_path=r"model\v1\output\PLM_FLOOR3.csv",location_vector_path=r'model\v1\output\location_vector_20m.csv'):
    path_load='data/processedData'+ os.sep + file_floor
    new_file_save='data/processedData' + os.sep + file_floor + os.sep + 'all_data.csv'
    if os.path.isdir(path_load):
        sf_files = os.listdir(path_load)
        all_data=pd.DataFrame() #某一层下所有SF的数据
        for sf_file in sf_files:
            sf_path=os.path.join(path_load,sf_file)
            if os.path.isdir(sf_path):
                csv_files=os.listdir(sf_path)
                for csv_file in csv_files:
                    if csv_file.endswith('.csv') and csv_file != 'all_data_new.csv' and csv_file != 'all_data.csv':
                        name = csv_file.replace('.csv', '')
                    temp_df=pd.read_csv(os.path.join(sf_path,csv_file))
                    temp_df['location_id'] = name
                    temp_df = temp_df[['realtime_average_rssi', 'average_rssi', 'rssi_variance', 'snr', 'median_rssi', 'mode_rssi', 'rssi_skewness', 'rssi_kurtosis', 'sf', 'tp', 'location_id']]
                    location_df=pd.read_csv(location_vector_path)
                    location_to_idx=dict(zip(location_df['location_id'], location_df['idx']))
                    location_to_distance_true=dict(zip(location_df['location_id'], location_df['distance_true']))
                    
                    temp_df['idx']=temp_df['location_id'].map(location_to_idx)
                    temp_df['Area']=temp_df['idx'].apply(lambda x: 1 if x in area1_list else (2 if x in area2_list else 3))
                    
                    params_df=pd.read_csv(PLM_params_path)
                    #merge比逐行查询效率高
                    merged_df=pd.merge(temp_df, params_df, on=['sf', 'Area'], how='left')
                    merged_df['distance_true'] = merged_df['location_id'].map(location_to_distance_true)
                    
                    tp=2    #后续修改这里的hardcode
                    merged_df['tp'] = tp
                    
                    valid_distances = merged_df['distance_true'] > 0
                    merged_df.loc[valid_distances, 'PLM_RSSI'] = merged_df.loc[valid_distances, 'A'] - 10 * merged_df.loc[valid_distances, 'n'] * np.log10(merged_df.loc[valid_distances, 'distance_true'])
                    merged_df.loc[~valid_distances, 'PLM_RSSI'] = np.nan
                    merged_df["residual"]= merged_df['realtime_average_rssi'] - merged_df['PLM_RSSI']
                    temp_df=merged_df[temp_df.columns.tolist() + ['PLM_RSSI',"residual"]]
                    
                    all_data=pd.concat([all_data,temp_df], ignore_index=True)
                    
                    all_data['location_id'] = all_data['location_id'].astype(str).str.replace('.0', '', regex=False)
                    all_data.loc[all_data['location_id'].str.isnumeric(), 'location_id'] = all_data.loc[all_data['location_id'].str.isnumeric(), 'location_id'].astype(int)    
        
        all_data=all_data.dropna()
        all_data=all_data.drop_duplicates()
        all_data.to_csv(new_file_save, index=False)


# 将all_data.csv转换为pth文件
def csv_to_pth(
    floor_id,
    pretrain_name='pretrain.pth',
    finetune_name='finetune.pth',
    test_name='test.pth',
    finger_name='finger.pth',
    pretrain_sf=[7,8,9,10,11,12],
    pretrain_label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    finetune_sf=[7,8,9,10,11,12],
    finetune_label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    test_sf=[7,8,9,10,11,12],
    test_label=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
    location_vector_path=r'model\v1\output\location_vector_20m.csv',
    max_pretrain=2000,
    max_finetune=500,
    max_test=200,
    times=1
):
    # 读取CSV数据
    csv_file = os.path.join(save_file_path, floor_id, 'all_data.csv')
    df = pd.read_csv(csv_file)

    # 读取location_vector
    location_df = pd.read_csv(location_vector_path)
    location_id_to_idx = dict(zip(location_df['location_id'], location_df['idx']))

    # 映射location_id
    df['location_id'] = df['location_id'].map(location_id_to_idx)
    df = df.dropna(subset=['location_id'])

    # 转换数据类型
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

    # orch
    # data_features = ['average_rssi','rssi_variance', 'median_rssi',"snr","mode_rssi"] * times
    # 我们的
    # data_features = ['average_rssi','rssi_variance', 'median_rssi', 'mode_rssi', "snr","residual"] * times
    
    # ab-residual 实验保证输入维度一样，进而保证网络的参数规模一致
    data_features = ['average_rssi','rssi_variance', 'median_rssi',"snr","mode_rssi","mode_rssi"] * times
    # 将df的data_features特征列进行归一化
    for feature in data_features:
        if feature in df.columns:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()+1e-3  # 防止除以0
    # 数据集划分
    pretrain_df = df[df['sf'].isin(pretrain_sf) & df['idx'].isin(pretrain_label)]
    pretrain_df = pretrain_df.groupby(['idx','sf'], group_keys=False).apply(lambda x: x.sample(n=min(len(x), max_pretrain), random_state=42))
    remaining_df = df.drop(pretrain_df.index)

    finetune_df = remaining_df[remaining_df['sf'].isin(finetune_sf) & remaining_df['idx'].isin(finetune_label)]
    finetune_df = finetune_df.groupby(['idx', 'sf'], group_keys=False).apply(lambda x: x.sample(n=min(len(x), max_finetune), random_state=42))
    remaining_df = remaining_df.drop(finetune_df.index)

    test_df = remaining_df[remaining_df['sf'].isin(test_sf) & remaining_df['idx'].isin(test_label)]
    test_df = test_df.groupby(['idx', 'sf'], group_keys=False).apply(lambda x: x.sample(n=min(len(x), max_test), random_state=42))
    remaining_df = remaining_df.drop(test_df.index)
    # 检查重叠
    assert len(set(pretrain_df.index) & set(finetune_df.index)) == 0
    assert len(set(pretrain_df.index) & set(test_df.index)) == 0
    assert len(set(finetune_df.index) & set(test_df.index)) == 0

    # 保存pth
    save_pth = 'model\\v1\\input'+os.sep+pretrain_name
    torch.save({
        'rssi': torch.tensor(pretrain_df[data_features].values, dtype=torch.float32),
        'sf': torch.tensor(pretrain_df["sf"].values, dtype=torch.float32)/10,
        'tp': torch.tensor(pretrain_df['tp'].values, dtype=torch.float32)/10,
        'snr': torch.tensor(pretrain_df[['sf', 'tp']].values, dtype=torch.float32)/10,
        'label': torch.tensor(pretrain_df['location_id'].values, dtype=torch.int64)
    }, save_pth)

    finetune_name='model\\v1\\input' + os.sep + finetune_name
    torch.save({
        'rssi': torch.tensor(finetune_df[data_features].values, dtype=torch.float32),
        'sf': torch.tensor(finetune_df["sf"].values, dtype=torch.float32)/10,
        'tp': torch.tensor(finetune_df['tp'].values, dtype=torch.float32)/10,
        'snr': torch.tensor(finetune_df[['sf', 'tp']].values, dtype=torch.float32)/10,
        'label': torch.tensor(finetune_df['location_id'].values, dtype=torch.int64)
    }, finetune_name)
    
    test_name='model\\v1\\input' + os.sep + test_name
    torch.save({
        'rssi': torch.tensor(test_df[data_features].values, dtype=torch.float32),
        'sf': torch.tensor(test_df["sf"].values, dtype=torch.float32)/10,
        'tp': torch.tensor(test_df['tp'].values, dtype=torch.float32)/10,
        'snr': torch.tensor(test_df[['sf', 'tp']].values, dtype=torch.float32)/10,
        'label': torch.tensor(test_df['location_id'].values, dtype=torch.int64)
    }, test_name)
    # finger数据集
    finger_pth_path = 'model\\v1\\input' + os.sep + finger_name
    torch.save({
        'features': torch.tensor(
            np.concatenate(
                [pretrain_df[data_features].values, (pretrain_df[['sf', 'tp']].values / 12)], axis=1
            ), dtype=torch.float32
        ),
        'label': torch.tensor(pretrain_df['location_id'].values, dtype=torch.int64)
    }, finger_pth_path)


if __name__ == "__main__":
    pretrain_pth_name = "floor3_sf_11_pretrain_dataset.pth"
    finetune_pth_name = "floor3_sf_10_finetune_dataset.pth"
    test_pth_name = "20m_sf9_floor3_test.pth"
    # txt_to_csv(f"FLOOR3")
    csv_to_csv(f"FLOOR3",PLM_params_path=r"model\v1\output\PLM_FLOOR3.csv")
    finetune_label=[0,1,3,4,5,7,9,11,13,15,17,19]
    csv_to_pth(f"FLOOR3",
               pretrain_name=pretrain_pth_name,
               pretrain_sf=[11],
                finetune_name=finetune_pth_name,
                finetune_sf=[9],
                finetune_label=[0,1,3,4,5,7,9,11,13,15,17,19],
                test_name=test_pth_name,
                test_sf=[9],
               finger_name="finger_sf_12_floor2_dataset.pth",
               max_pretrain=0,max_finetune=0,max_test=700)
    # rssi = pretrain_pth['rssi']
    # for i in range(rssi.shape[1]):
    #     print(f"Dimension {i}: min={rssi[:, i].min().item()}, max={rssi[:, i].max().item()}")
    #     # 计算方差
    #     print(f"Dimension {i}: variance={rssi[:, i].var().item()}")
    # for i in range(pretrain_pth['snr'].shape[1]):
    #     print(f"snr Dimension {i}: min={pretrain_pth['snr'][:, i].min().item()}, max={pretrain_pth['snr'][:, i].max().item()}")
    
    pretrain_pth=torch.load('model\\v1\\input\\'+pretrain_pth_name)
    test_pth=torch.load('model\\v1\\input\\'+test_pth_name)
    finetune_pth=torch.load('model\\v1\\input\\'+finetune_pth_name)
    print("pretrain label shape: ",pretrain_pth['label'].shape)
    print("pretrain label unique: ", pretrain_pth['label'].unique())
    print("finetune rssi shape: ",finetune_pth['rssi'].shape)
    print("finetune label unique: ",finetune_pth['label'].unique())
    print("test label shape: ",test_pth['label'].shape)
    print("test label unique: ", test_pth['label'].unique())
    print("pretrain sf unique values: ",pretrain_pth['sf'].unique())
    print("finetune sf unique values: ",finetune_pth['sf'].unique())
    print("test sf unique values: ",test_pth['sf'].unique())  
    print("finetune snr unique values: ",finetune_pth['snr'].unique())
    input("Press Enter to exit...")  # 等待用户输入以查看输出
