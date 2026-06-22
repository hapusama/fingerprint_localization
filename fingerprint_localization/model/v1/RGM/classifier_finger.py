import numpy as np
from sklearn.base import defaultdict
from sklearn.metrics import accuracy_score
import torch
import os
import json
import pandas as pd
import torch.nn as nn
from src.parameter_paser import parse_args_finetune
import torch.optim as optim
from src.dataset import generate_three_loader_v3,generate_three_loader_v2
from src.dataset import generate_three_dataset_v2, ComplexDatasetLocs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score,classification_report
# 模型配置

batch_size = 128
input_dim = None
# mode = 'knn'
mode='generate'
# mode='test'
# mode='original'
# mode="ori"
num_classes = None
# 批次的大小
lr = 1e-2
# 优化器的学习率
valid_size = 0.2
test_size=0.2
num_epochs = 100
# num_epochs=170
new_path = r'd:\Desktop\PHD\research\biyework\maml'
ori_pth = r"model\v1\input\floor3_sf_9_test_dataset.pth"
location_vector_path = r"model\v1\output\location_vector_v2.csv"
# location_vector_path = r"model\v1\output\location_vector_5m.csv"
# location_vector_path = r"model\v1\output\location_vector_20m.csv"

area1_list=[0,1,2,3,4,5]
area2_list=[6,7,8,9,10,11,12,13,14]
area3_list=[15,16,17,18,19,20]
from tqdm import tqdm
import time
from torchinfo import summary
import psutil
# 3. 构建模型
class LocationClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LocationClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
        
if "__main__"==__name__:
    base_dir = os.path.dirname(os.path.realpath(__file__))
    args=parse_args_finetune()

    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    input_data_pth=os.path.join(output_dir,args.data_name_fake)
    print("input_data_pth: ", input_data_pth)
    model_path_train=os.path.join(output_dir,args.save_model_name_fake)
    print(f"model_path_train: {model_path_train}")

    complex_dataset_generated_real=torch.load(input_data_pth)
    sample_fake, sample_real, _, _ = complex_dataset_generated_real[0]
    input_dim = sample_fake.squeeze().numel()
    labels_for_dim = [
        int(complex_dataset_generated_real[i][3])
        for i in range(len(complex_dataset_generated_real))
    ]
    num_classes = max(getattr(args, "num_locs", 0), max(labels_for_dim) + 1)
    print(f"classifier input_dim: {input_dim}, num_classes: {num_classes}")
    
    train_loader,valid_loader,test_loader=generate_three_loader_v3(complex_dataset_generated_real, 
                                                                   batch_size, 
                                                                   valid_size, 
                                                                   test_size)

    print(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}")
    # 初始化模型
    model = LocationClassifier(input_dim, num_classes)
    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 应用学习率下降策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.1, verbose=True)
    # 设置随机数种子

    # 5. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if mode == "test":
        # 加载已训练模型
        model.load_state_dict(torch.load(model_path_train))
        model.eval()

        # 随便取一条测试数据
        for batch_idx, (_, data_batch_real, _, label_int_batch) in enumerate(test_loader):
            data_batch_real = data_batch_real.to(device)
            data_batch_real = data_batch_real.squeeze(1)
            # 只取一条
            single_data = data_batch_real[0].unsqueeze(0)
            break
            # 推理计时和计算开销
        with torch.no_grad():
            start_time = time.time()
            output = model(single_data)
            _, predicted = torch.max(output.data, 1)
            end_time = time.time()
            infer_time = end_time - start_time
            # 计算推理时的 FLOPs 和参数量
            model_summary = summary(model, input_size=single_data.shape, verbose=0)
            flops = model_summary.total_mult_adds
            params = model_summary.total_params
            print(f"Inference time for one sample: {infer_time * 1000:.4f} ms")
            print(f"Predicted label: {predicted.item()}")
            print(f"Model FLOPs (per sample): {flops}")
            print(f"Model total parameters: {params}")
            # 内存开销
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"Memory usage (RSS): {mem_info.rss / 1024 ** 2:.2f} MB")
    if mode == 'knn':
        data_pth = os.path.join(input_dir , "knn_sf11_floor3_dataset.pth")
        loaded = torch.load(data_pth)
        rssi = loaded['rssi']  # shape [batch_size,4]
        snr = loaded['snr']
        label = loaded['label'] # shape [batch_size,]
        # 使用KNN进行分类

        knn = KNeighborsClassifier(n_neighbors=5)
        X = np.concatenate([rssi, snr], axis=1)
        y = label
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        print(f"KNN Test Accuracy: {accuracy:.4f}")
        print(f"KNN Test Recall (macro): {recall:.4f}")
        print(f"KNN Test Precision (macro): {precision:.4f}")
        print(classification_report(y_test, y_pred))
        input("Enter to exit...")  # 等待用户输入以查看输出
        
    if mode=='generate':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx, (data_batch_fake, _, _, label_int_batch) in enumerate(tqdm(train_loader)):
                data_batch_fake, label_int_batch = data_batch_fake.to(device), label_int_batch.to(device)
                data_batch_fake = data_batch_fake.squeeze(1)
                # 在训练循环中添加噪声
                data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                # 随机丢弃部分特征
                dropout_mask = torch.rand_like(data_batch_fake) > 0.1  # 90% 的概率保留特征
                data_batch_fake *= dropout_mask
                optimizer.zero_grad()
                outputs = model(data_batch_fake)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(valid_loader):
                    data_batch_fake,data_batch_real, label_int_batch = data_batch_fake.to(device), data_batch_real.to(device), label_int_batch.to(device)
                    data_batch_real = data_batch_real.squeeze(1)
                    data_batch_fake = data_batch_fake.squeeze(1)
                    # 在训练循环中添加噪声
                    data_batch_fake += torch.randn_like(data_batch_fake) * 0.01  # 添加高斯噪声
                    # 随机丢弃部分特征
                    dropout_mask = torch.rand_like(data_batch_fake) > 0.1  # 90% 的概率保留特征
                    data_batch_fake *= dropout_mask
                    # 在验证循环中添加噪声
                    data_batch_real = data_batch_real.squeeze(1)
                    data_batch_real += torch.randn_like(data_batch_real) * 0.01  # 添加高斯噪声
                    # 随机丢弃部分特征
                    dropout_mask = torch.rand_like(data_batch_real) > 0.1  # 90% 的概率保留特征
                    data_batch_real *= dropout_mask
                    # data_batch_fake = data_batch_fake.view(-1, input_dim)
                    outputs = model(data_batch_fake)
                    loss = criterion(outputs, label_int_batch)
                    valid_loss += loss.item()
                    
            valid_loss /= len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
        # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        location_vector_df = pd.read_csv(location_vector_path)
        coords = location_vector_df.set_index('idx')[['true_x', 'true_y']].to_dict('index')
        adjacent_list = list(coords.keys())
        with torch.no_grad():
            label_correct = defaultdict(int)
            label_total = defaultdict(int)
            label_pred_total = defaultdict(int)
            total_distance_error = 0.0
            unmatched_count = 0
            label_int_batch_total = []
            predicted_total = []
            for batch_idx, (_, data_batch_real, _, label_int_batch) in enumerate(test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                
                label_int_batch_total.extend(label_int_batch.cpu().numpy())
                predicted_total.extend(predicted.cpu().numpy())
                for true_label, pred_label in zip(label_int_batch.cpu().numpy(), predicted.cpu().numpy()):
                    label_total[true_label] += 1
                    label_pred_total[pred_label] += 1
                    if true_label == pred_label:
                        label_correct[true_label] += 1
                    else:
                        # 计算未匹配到的定位误差
                        if true_label in coords and pred_label in coords:
                            x1, y1 = coords[true_label]['true_x'], coords[true_label]['true_y']
                            x2, y2 = coords[pred_label]['true_x'], coords[pred_label]['true_y']
                            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                            total_distance_error += dist
                            unmatched_count += 1
                total += label_int_batch.size(0)
                correct += (predicted == label_int_batch).sum().item()
            if unmatched_count > 0:
                avg_distance_error = total_distance_error / total
                print(f"Average localization error for unmatched labels: {avg_distance_error:.4f}")
            else:
                print("All labels matched, no localization error for unmatched labels.")
            accuracy = correct / total
            recall = recall_score(label_int_batch_total, predicted_total, average='macro')
            precision = precision_score(label_int_batch_total, predicted_total, average='macro')
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test Precision: {precision:.4f}")
            # 统计每个区域的 recall、accuracy、precision
            area_metrics = {}
            for area_name, area_list in zip(['area1', 'area2', 'area3'], [area1_list, area2_list, area3_list]):
                # 筛选属于该区域的标签
                area_mask = np.isin(label_int_batch_total, area_list)
                area_true = np.array(label_int_batch_total)[area_mask]
                area_pred = np.array(predicted_total)[area_mask]
                # accuracy
                area_acc = np.mean(area_true == area_pred) if len(area_true) > 0 else 0
                # recall
                area_recall = recall_score(area_true, area_pred, labels=area_list, average='macro', zero_division=0) if len(area_true) > 0 else 0
                # precision
                area_precision = precision_score(area_true, area_pred, labels=area_list, average='macro', zero_division=0) if len(area_true) > 0 else 0
                area_metrics[area_name] = {
                    'accuracy': area_acc,
                    'recall': area_recall,
                    'precision': area_precision
                }
                print(f"{area_name} - Accuracy: {area_acc:.4f}, Recall: {area_recall:.4f}, Precision: {area_precision:.4f}")
            print(classification_report(label_int_batch_total, predicted_total))
            
            # 统计距离网关不同distance范围的点的匹配结果
            # 按照 distance_true 划分点的分组
            distance_bins = [0, 15, 30, 45, 60, 75, np.inf]
            distance_labels = ['0-15', '15-30', '30-45', '45-60', '60-75', '75+']
            location_vector_df = pd.read_csv(location_vector_path)
            location_vector_df['distance_bin'] = pd.cut(location_vector_df['distance_true'], bins=distance_bins, labels=distance_labels, right=False)

            # 构建标签到距离分组的映射
            label_to_distance_bin = location_vector_df.set_index('idx')['distance_bin'].to_dict()

            # 收集每个分组的真实标签和预测标签
            bin_true_labels = {bin_label: [] for bin_label in distance_labels}
            bin_pred_labels = {bin_label: [] for bin_label in distance_labels}

            for true_label, pred_label in zip(label_int_batch_total, predicted_total):
                bin_label = label_to_distance_bin.get(true_label, None)
                if bin_label is not None:
                    bin_true_labels[bin_label].append(true_label)
                    bin_pred_labels[bin_label].append(pred_label)

            # 统计每个分组的 recall 和 precision
            for bin_label in distance_labels:
                true = np.array(bin_true_labels[bin_label])
                pred = np.array(bin_pred_labels[bin_label])
                if len(true) > 0:
                    bin_recall = recall_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)
                    bin_precision = precision_score(true, pred, labels=np.unique(true), average='macro', zero_division=0)
                    bin_acc = np.mean(true == pred)
                    print(f"Distance bin {bin_label}: Accuracy={bin_acc:.4f}, Recall={bin_recall:.4f}, Precision={bin_precision:.4f}, Count={len(true)}")
                else:
                    print(f"Distance bin {bin_label}: No samples.")
        labels = sorted(set(list(label_total.keys()) + list(label_pred_total.keys())))
        accuracies = [label_correct[l] / label_total[l] if label_total[l] > 0 else 0 for l in labels]
        recalls = [label_correct[l] / label_pred_total[l] if label_pred_total[l] > 0 else 0 for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        plt.bar(x + width/2, recalls, width, label='Recall', color='orange')
        plt.xlabel('Label')
        plt.ylabel('Score')
        plt.title('Per-label Accuracy and Recall')
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        metrics_out = os.path.splitext(model_path_train)[0] + "_metrics.json"
        plot_out = os.path.splitext(model_path_train)[0] + "_per_label.png"
        metrics_payload = {
            "accuracy": float(accuracy),
            "recall": float(recall),
            "precision": float(precision),
            "average_localization_error": float(avg_distance_error) if unmatched_count > 0 else 0.0,
            "area_metrics": area_metrics,
            "labels": [int(label) for label in labels],
            "per_label_accuracy": [float(value) for value in accuracies],
            "per_label_recall": [float(value) for value in recalls],
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
        }
        with open(metrics_out, "w", encoding="utf-8") as fp:
            json.dump(metrics_payload, fp, indent=2, ensure_ascii=False)
        plt.savefig(plot_out, dpi=160)
        plt.close()
        print(f"Saved classifier metrics to {metrics_out}")
        print(f"Saved per-label plot to {plot_out}")

    if mode=='original':
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(tqdm(train_loader)):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                optimizer.zero_grad()
                outputs = model(data_batch_real)
                loss = criterion(outputs, label_int_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(data_batch_fake,data_batch_real,_,label_int_batch) in enumerate(valid_loader):
                    data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                    data_batch_real = data_batch_real.squeeze(1)
                    # data_batch_fake = data_batch_fake.view(-1, input_dim)
                    outputs = model(data_batch_real)
                    loss = criterion(outputs, label_int_batch)
                    valid_loss += loss.item()
                    
            valid_loss /= len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
        # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx,(_,data_batch_real,_,label_int_batch) in enumerate(test_loader):
                data_batch_real, label_int_batch = data_batch_real.to(device), label_int_batch.to(device)
                data_batch_real = data_batch_real.squeeze(1)
                # data_batch_real = data_batch_real.view(-1, input_dim)
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                total += label_int_batch.size(0)
                # 把匹配成功的点打印出来
                matched_labels = label_int_batch[predicted == label_int_batch]
                print(f"Matched Labels: {matched_labels.cpu().numpy()}")
                
                correct += (predicted == label_int_batch).sum().item()
        accuracy = correct / total
        recall = correct / (total - unmatched_count) if (total - unmatched_count) > 0 else 0
        
        print(f"Test Accuracy: {accuracy:.4f}")
    if mode=="ori" :
        loaded_data = torch.load(ori_pth)
        rssi = loaded_data['rssi']  
        snr = loaded_data['snr']
        label = loaded_data['label']
        
        complex_dataset = ComplexDatasetLocs(rssi, 
                                            snr, 
                                            label, 
                                            location_vector_path
                                            )
        train_set,valid_set,test_set=generate_three_dataset_v2(complex_dataset, 
                                                                    valid_size, 
                                                                    test_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        for epoch in range(num_epochs):
            total_loss = 0
            for (input_vec,_,label) in tqdm(train_loader):
                input_vec = input_vec.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                outputs = model(input_vec)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx,(input_vec,_,label) in enumerate(valid_loader):
                    input_vec = input_vec.to(device)
                    label = label.to(device)
                    outputs = model(input_vec)
                    loss = criterion(outputs, label)
                    valid_loss += loss.item()
                    
            valid_loss /= len(valid_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f},Validation Loss: {valid_loss:.4f}")
            # 更新学习率
            scheduler.step(valid_loss)
            # 如果验证损失没有改善，则保存当前模型
            torch.save(model.state_dict(), model_path_train)
        # 加载模型
        model.load_state_dict(torch.load(model_path_train))
        # 6. 测试模型
        model.eval()
        correct = 0
        total = 0
        location_vector_df = pd.read_csv(location_vector_path)
        coords = location_vector_df.set_index('idx')[['true_x', 'true_y']].to_dict('index')
        adjacent_list = list(coords.keys())
        with torch.no_grad():
            label_correct = defaultdict(int)
            label_total = defaultdict(int)
            label_pred_total = defaultdict(int)
            total_distance_error = 0.0
            unmatched_count = 0
            # 收集所有测试集标签和预测结果
            all_true_labels = []
            all_pred_labels = []
            for batch_idx,(input_vec,_,label) in enumerate(test_loader):
                input_vec, label = input_vec.to(device), label.to(device)
                # input_vec = input_vec.view(-1, input_dim)
                outputs = model(input_vec)
                _, predicted = torch.max(outputs.data, 1)
                for true_label, pred_label in zip(label.cpu().numpy(), predicted.cpu().numpy()):
                    label_total[true_label] += 1
                    label_pred_total[pred_label] += 1
                    if true_label == pred_label:
                        label_correct[true_label] += 1
                    else:
                        # 计算未匹配到的定位误差
                        if true_label in coords and pred_label in coords:
                            x1, y1 = coords[true_label]['true_x'], coords[true_label]['true_y']
                            x2, y2 = coords[pred_label]['true_x'], coords[pred_label]['true_y']
                            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                            total_distance_error += dist
                            unmatched_count += 1
                total += label.size(0)
                all_true_labels.extend(label.cpu().numpy())
                all_pred_labels.extend(predicted.cpu().numpy())
                correct += (predicted == label).sum().item()
            if unmatched_count > 0:
                avg_distance_error = total_distance_error / total
                print(f"Average localization error for unmatched labels: {avg_distance_error:.4f}")
            else:
                print("All labels matched, no localization error for unmatched labels.")
            accuracy = correct / total

            recall = recall_score(all_true_labels, all_pred_labels, average='macro')
            precision = precision_score(all_true_labels, all_pred_labels, average='macro')
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(classification_report(label.cpu().numpy(), predicted.cpu().numpy()))
        labels = sorted(set(list(label_total.keys()) + list(label_pred_total.keys())))
        accuracies = [label_correct[l] / label_total[l] if label_total[l] > 0 else 0 for l in labels]
        recalls = [label_correct[l] / label_pred_total[l] if label_pred_total[l] > 0 else 0 for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        plt.bar(x + width/2, recalls, width, label='Recall', color='orange')
        plt.xlabel('Label')
        plt.ylabel('Score')
        plt.title('Per-label Accuracy and Recall')
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
