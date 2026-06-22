import os
import matplotlib.pyplot as plt

import pandas as pd
from classifier_finger import *
from src.parameter_paser import parse_args_finetune
import numpy as np
from collections import Counter
import random
adjacent_list = [0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20]
coords = []
rollback_count = 0
np.random.seed(42)
random.seed(42)
def find_nearest_point(pred_x, pred_y, topk_labels=None, topk_probs=None, prob_threshold=0.1):
    """
    找到距离预测位置最近的点
    :param pred_x: 预测的x坐标
    :param pred_y: 预测的y坐标
    :param topk_labels: 可选的top-k标签列表，如果提供则只在这些点中寻找
    :param topk_probs: 对应的概率列表
    :param prob_threshold: 概率阈值，低于此值的点将被忽略
    :return: 最近点的索引
    """
    min_dist = float('inf')
    nearest_idx = None
    
    # 如果提供了topk_labels，则在其中寻找
    if topk_labels is not None and topk_probs is not None:
        for label, prob in zip(topk_labels, topk_probs):
            if prob < prob_threshold:
                continue
            label = int(label)
            if label not in coords:
                continue
            d = np.sqrt((coords[label]['true_x'] - pred_x)**2 + 
                       (coords[label]['true_y'] - pred_y)**2)
            if d < min_dist:
                min_dist = d
                nearest_idx = label
    else:
        # 否则在所有点中寻找
        for idx in adjacent_list:
            d = np.sqrt((coords[idx]['true_x'] - pred_x)**2 + 
                       (coords[idx]['true_y'] - pred_y)**2)
            if d < min_dist:
                min_dist = d
                nearest_idx = idx
    
    return nearest_idx

def vector_extrapolation_prediction(predicted_route):
    """
    使用向量外推方法预测下一个点
    :param predicted_route: 已预测的路径
    :return: 预测的下一个点索引，如果无法预测则返回None
    """
    if len(predicted_route) < 2:
        return None
        
    # 获取最后两个点
    last_point = predicted_route[-1]
    last_2_point = predicted_route[-2]
    
    # 计算向量
    x1, y1 = coords[last_2_point]['true_x'], coords[last_2_point]['true_y']
    x2, y2 = coords[last_point]['true_x'], coords[last_point]['true_y']
    dx, dy = x2 - x1, y2 - y1
    
    # 用同样的方向和距离从last_point出发
    pred_x = x2 + dx
    pred_y = y2 + dy
    
    # 找到最近的点
    return find_nearest_point(pred_x, pred_y)
# 计算两点之间的距离
def distance(a, b):
    # 如果a和b相等，距离为0
    if a == b:
        return 0.0
    # 找到a和b在adjacent_list中的索引
    try:
        idx_a = adjacent_list.index(a)
        idx_b = adjacent_list.index(b)
    except ValueError:
        # 如果a或b不在adjacent_list中，返回inf
        return float('inf')
    # 确保从小到大遍历
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    dist = 0.0
    for i in range(idx_a, idx_b):
        p1 = adjacent_list[i]
        p2 = adjacent_list[i + 1]
        ax, ay = coords[p1]['true_x'], coords[p1]['true_y']
        bx, by = coords[p2]['true_x'], coords[p2]['true_y']
        dist += np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    return dist

# 随机生成一条路线，点可重复，但每个点最多出现两次，总点数不超过max_length，且相邻点距离不超过max_dist米
def generate_route(points, max_dist=17, max_length=20):
    # 每个点最多出现两次，必须覆盖所有点至少一次
    remaining = set(points)
    route = [np.random.choice(points)]
    remaining.discard(route[-1])
    counter = Counter(route)
    while (remaining or any(counter[p] < 2 for p in points)) and len(route) < max_length:
        # 只在未超出出现次数限制的点中选，且距离不超过max_dist
        candidates = [p for p in points if counter[p] < 2 and distance(route[-1], p) <= max_dist]
        if not candidates:
            break
        next_point = np.random.choice(candidates)
        route.append(next_point)
        counter[next_point] += 1
        remaining.discard(next_point)
    return route

if __name__=="__main__":
    args = parse_args_finetune()
    save_model_name_fake = args.save_model_name_fake
    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = args.input_dim + 2  # 加2是因为有两个额外的坐标维度
    num_classes = args.num_locs
    finetune_data_name = args.data_name
    finetune_data_path = os.path.join(input_dir , finetune_data_name)
    loaded_data = torch.load(finetune_data_path)
    rssi = loaded_data['rssi']
    sf_tp = loaded_data['snr']
    label = loaded_data['label']
    # 将RSSI和sf_tp拼接
    input_data = np.concatenate((rssi, sf_tp), axis=1)
    
    model_check_point_pth = os.path.join(output_dir, save_model_name_fake)
    print(f"model_check_point_pth: {model_check_point_pth}")
    model = LocationClassifier(input_dim, num_classes)
    
    # load the model
    model.load_state_dict(torch.load(model_check_point_pth, map_location=device))
    model.to(device)
    model.eval()
    
    location_vector = os.path.join(output_dir, args.location_vector_name)
    location_vector = pd.read_csv(location_vector)
    dataset_path = os.path.join(output_dir, args.data_name_fake)
    
    loaded_data = torch.load(dataset_path)
    
    # 获取所有点的坐标
    coords = location_vector.set_index('idx')[['true_x', 'true_y']].to_dict('index')
    # 计算每个相邻点之间的平均距离
    avg_distances = []
    for i in range(len(adjacent_list) - 1):
        dist = distance(adjacent_list[i], adjacent_list[i + 1])
        avg_distances.append(dist)
    print("Average distances between adjacent points:", np.mean(avg_distances))
    print(distance(17,9))
    input_data_list = []
    # 生成一条合法路线

    route_list =[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20]]
    generate_route= []
    for route in route_list:
        # route = generate_route(adjacent_list)
        route_data = []
        for point in route:
            # 找到所有label等于point的索引
            indices = np.where(label == point)[0]
            if len(indices) == 0:
                continue  # 如果没有对应的input data则跳过
            # 随机选择一个索引
            chosen_idx = np.random.choice(indices)
            # 取出对应的input data
            generate_route.append(int(label[chosen_idx]))
            route_data.append(input_data[chosen_idx])
        # 将route data中的每一条数据输入model
        predicted_route = []
        with torch.no_grad():
            for data in route_data:
                last_point = predicted_route[-1] if predicted_route else None
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(data_tensor)
                probs = torch.softmax(output, dim=1)
                topk_probs, topk_labels = torch.topk(probs, k=probs.shape[1], dim=1)

                topk_labels = topk_labels.view(-1).cpu().numpy()
                topk_probs = topk_probs.view(-1).cpu().numpy()
                chosen_label = topk_labels[0]
                chosen_label = int(chosen_label)
                
                # 纠错模块
                last_point_now = predicted_route[-1] if predicted_route else None
                
                # # 记录回滚次数
                # if 'rollback_count' not in locals():
                #     rollback_count = 0
                
                # # 检测是否需要纠错（超过阈值 or 置信度过低）
                # need_correction = (last_point_now is not None and distance(last_point_now, chosen_label) > 30) or topk_probs[0] < 0.3
                
                # if need_correction:
                #     # rollback_count += 1
                #     # if rollback_count > 3:
                #     #     print("Rollback exceeded 3 times, stopping prediction.")
                #     #     break

                #     # 方案2: 向量外推预测
                #     vector_pred = vector_extrapolation_prediction(predicted_route)
                #     # 选择纠错方案
                #     if vector_pred is not None:
                #         dist_vector = distance(last_point_now, vector_pred)
                #         chosen_label = vector_pred
                #         print(f"Using vector prediction: {vector_pred} to replace {last_point_now} in Step {len(predicted_route) + 1} and true label is {route[len(predicted_route) - 1]}")
                #     else:
                #         # 如果两种方案都不可用，直接回滚
                #         chosen_label = last_point_now
                #         print("No correction method available, rolling back.")
                        
                predicted_route.append(chosen_label)
                # 对比真实点和预测点，若不匹配则输出概率分布
                if len(predicted_route) <= len(route):
                    true_point = route[len(predicted_route) - 1]
                    if chosen_label != true_point:
                        print(f"Step {len(predicted_route)}: True={true_point}, Pred={chosen_label}")
                        top3 = list(zip(topk_labels, topk_probs))[:3]
                        print(f"Top 3 Probabilities: {dict(top3)}")
        print("Generated route:", generate_route)
        print("Predicted route:", predicted_route)
        # 计算accuracy
        correct = sum([p == t for p, t in zip(predicted_route, generate_route)])
        # 计算MLE（平均定位误差）
        if len(predicted_route) == len(generate_route):
            total_dist = sum([distance(p, t) for p, t in zip(predicted_route, generate_route)])
            mle = total_dist / len(generate_route) if len(generate_route) > 0 else 0
            print("MLE (Mean Localization Error):", mle)
        else:
            print("Cannot compute MLE: route lengths do not match.")
        accuracy = correct / len(generate_route) if len(generate_route) > 0 else 0
        print("accuracy: ", accuracy)
        input("Press Enter to continue...")
        