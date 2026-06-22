import numpy as np
import os
import pandas as pd
from src.parameter_paser import parse_args_location
# 350和354之间的距离是9m

# 返回一个向量 (x_direction, y_direction, distance)，表示节点和网关之间的方向
def calculate_location_vector(node_coord, gateway_coord, meters_per_pixel=1.0):
    """
    计算节点与网关之间的向量信息。

    返回值包含：
      - x_direction, y_direction: 单位方向向量（基于像素坐标归一化）
      - distance_pixels: 像素距离（原始坐标单位）
      - distance_true: 真实世界距离（米，使用 meters_per_pixel 缩放）
      - true_x, true_y: 节点相对于网关的真实世界坐标差（米）

    meters_per_pixel: 每个像素长度对应的米数（m / pixel）。
    """
    # 方向从节点指向网关，用于描述节点与网关之间的相对方位（像素座标）
    node = np.array(node_coord, dtype=float)
    gateway = np.array(gateway_coord, dtype=float)
    direction_vector = gateway - node
    distance_pixels = float(np.linalg.norm(direction_vector))

    if distance_pixels == 0:
        unit_vector = np.zeros_like(direction_vector)
    else:
        unit_vector = direction_vector / distance_pixels

    # distance_true: 节点与网关之间的真实距离（米）
    distance_true = float(distance_pixels * float(meters_per_pixel))

    # true_x / true_y: 将节点在像素坐标系中的 (x,y) 转为真实世界坐标（米）
    # 注意：这里的原点与像素坐标的原点一致（例如左上角），仅按比例尺转换
    true_x = float(node[0] * float(meters_per_pixel))
    true_y = float(node[1] * float(meters_per_pixel))

    location_vector = [float(unit_vector[0]), float(unit_vector[1]), distance_pixels, distance_true, true_x, true_y]
    return location_vector

def calculate_distance_true(node_coord, gateway_coord):
    return np.linalg.norm(np.array(gateway_coord) - np.array(node_coord))

if __name__ == '__main__':
    args = parse_args_location()
    print(f"\nUsing configuration file: {args.configs}\n")

    output_dir = r"model\v1\output"
    os.makedirs(output_dir, exist_ok=True)

    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    # 网关坐标
    gateway_coord_m = (args.gateway_X, args.gateway_Y)
    # 遍历 label_coordinate.csv 文件，计算节点和网关之间的方向向量
    df = pd.read_csv(r"model\v1\input\label_coordinate.csv")
    
    
    # 先从 df 中用已知点（id==50 和 id==54）计算像素->米的比例
    meters_between_calib = 9.3  # 已知这两个点之间的真实距离（米）
    # 注意：CSV 中的 id 列有字符串和数字混合，确保按字符串匹配 '50' 和 '54'
    calib_row_50 = df[df['id'].astype(str) == '50']
    calib_row_54 = df[df['id'].astype(str) == '54']
    if not calib_row_50.empty and not calib_row_54.empty:
        x50, y50 = float(calib_row_50.iloc[0]['x']), float(calib_row_50.iloc[0]['y'])
        x54, y54 = float(calib_row_54.iloc[0]['x']), float(calib_row_54.iloc[0]['y'])
        pixel_dist = np.linalg.norm(np.array([x50 - x54, y50 - y54], dtype=float))
        if pixel_dist > 0:
            meters_per_pixel = meters_between_calib / float(pixel_dist)
            print(f"Calibration: pixel_dist={pixel_dist:.2f}, meters_per_pixel={meters_per_pixel:.6f}")
        else:
            meters_per_pixel = 1.0
            print("Warning: calibration points have zero pixel distance; using meters_per_pixel=1.0")
    else:
        meters_per_pixel = 1.0
        print("Warning: calibration ids '50' or '54' not found; using meters_per_pixel=1.0")

    # 逐行构建输出表（先收集像素距离，以便统一归一化到 distance 列）
    rows = []
    idx = 0
    for index, row in df.iterrows():
        node_id = row['id']
        node_coord = (row['x'], row['y'])
        if node_coord == gateway_coord_m:
            continue
        loc_vec = calculate_location_vector(node_coord, gateway_coord_m, meters_per_pixel=meters_per_pixel)
        # loc_vec: [x_dir, y_dir, distance_pixels, distance_true, true_x, true_y]
        # 保存 x,y 为从节点指向网关的单位方向向量（归一化）
        rows.append({
            'location_id': node_id,
            'x': loc_vec[0],  # unit x direction (gateway - node) normalized
            'y': loc_vec[1],  # unit y direction
            'distance_pixels': loc_vec[2],
            'distance_true': loc_vec[3],
            'true_x': loc_vec[4],  # node x in meters (pixel->m)
            'true_y': loc_vec[5],  # node y in meters
            'idx': idx
        })
        idx += 1

    out_df = pd.DataFrame(rows)
    # 将 distance_pixels 列归一化为 distance（保持原来脚本对 distance 的期望含义）
    if not out_df.empty:
        out_df['distance'] = (out_df['distance_pixels'] - out_df['distance_pixels'].mean()) / out_df['distance_pixels'].std()
    else:
        out_df['distance'] = []

    # 最终列顺序：location_id,x,y,distance,distance_true,true_x,true_y,idx
    final_cols = ['location_id', 'x', 'y', 'distance', 'distance_true', 'true_x', 'true_y', 'idx']
    out_df = out_df[final_cols]
    out_df.to_csv(location_vector_path, index=False)
    
    # 确保文件已经写入成功后再读取
    # 计算所有点之间的平均间隔（按 idx 顺序相邻）
    
    # # 读取 location_vector 文件
    # out_df = pd.read_csv(location_vector_path)
    # out_df = out_df.sort_values(by='idx').reset_index(drop=True)
    if len(out_df) > 1:
        coords = out_df[['true_x', 'true_y']].values
        diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        avg_interval = np.mean(diffs)
        print(f"Average interval between adjacent points: {avg_interval:.4f} meters")
    else:
        print("Not enough points to calculate average interval.")