import numpy as np
from sklearn.base import defaultdict
from sklearn.metrics import accuracy_score
import torch
import os
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
from sklearn.metrics import accuracy_score, recall_score, precision_score
# 模型配置
batch_size = 128
input_dim=8
# mode = 'knn'
mode='generate'
# mode='test'
# mode='original'
# mode="ori"
num_classes=21
# 批次的大小
lr = 1e-2
# 优化器的学习率
valid_size = 0.05
test_size=0.25
num_epochs = 150
# num_epochs=250
new_path = r'd:\Desktop\PHD\research\biyework\maml'
ori_pth = r"model\v1\input\floor3_sf_11_pretrain_dataset.pth"
location_vector_path = r"model\v1\output\location_vector_v2.csv"
from tqdm import tqdm
from thop import profile
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
    input_data_pth=os.path.join(output_dir,args.data_name_fake)
    print("input_data_pth: ", input_data_pth)
    model_path_train=os.path.join(output_dir,args.save_model_name_fake)
    print(f"model_path_train: {model_path_train}")

    complex_dataset_generated_real=torch.load(input_data_pth)
    
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
        # 根据定位误差绘制CDF图
        # 读取定位点坐标
        location_vector_df = pd.read_csv(location_vector_path)
        coords = location_vector_df.set_index('idx')[['true_x', 'true_y']].to_dict('index')

        # 计算定位误差
        distance_errors = []
        for true_label, pred_label in zip(y_test, y_pred):            
            true_label = int(true_label)
            pred_label = int(pred_label)
            x1, y1 = coords[true_label]['true_x'], coords[true_label]['true_y']
            x2, y2 = coords[pred_label]['true_x'], coords[pred_label]['true_y']
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            distance_errors.append(dist)

        if len(distance_errors) > 0:
            avg_error = np.mean(distance_errors)
            print(f"Average Localization Error: {avg_error:.4f} meters")
            sorted_errors = np.sort(distance_errors)
            errors_df = pd.DataFrame({
                'KNN Localization Error': sorted_errors
            })
            save_csv_pth = os.path.join(output_dir,"knn_localization_error.csv")
            errors_df.to_csv(save_csv_pth, index=False)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            # 保存到CSV
            # cdf_df = pd.DataFrame({
            # 'error': sorted_errors,
            # 'cdf': cdf
            # })
            # cdf_csv_path = os.path.join(output_dir, 'knn_localization_error_cdf.csv')
            # cdf_df.to_csv(cdf_csv_path, index=False)
            # print(f"CDF data saved to {cdf_csv_path}")
            plt.figure()
            plt.plot(sorted_errors, cdf, marker='.', linestyle='-')
            plt.xlabel('Localization Error (meters)')
            plt.ylabel('CDF')
            plt.title('CDF of Localization Error (KNN)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No localization errors to plot CDF.")
        # print(classification_report(y_test, y_pred))
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
            mse_loss = nn.MSELoss()
            total_mse = 0.0
            num_batches = 0

            for batch_idx, (data_batch_fake, data_batch_real, _, label_int_batch) in enumerate(test_loader):
                data_batch_fake, data_batch_real, label_int_batch = data_batch_fake.to(device), data_batch_real.to(device), label_int_batch.to(device)
                data_batch_fake = data_batch_fake.squeeze(1)
                data_batch_real = data_batch_real.squeeze(1)
                # 计算real的MSE（自身，应该为0）
                mse = mse_loss(data_batch_real, data_batch_fake)
                total_mse += mse.item()
                num_batches += 1
                # 评估系统开销：模型参数量和FLOPs
                macs, params = profile(model, inputs=(data_batch_real,))
                print(f"Model FLOPs (MACs): {macs}, Parameters: {params}, data_batch_size: {data_batch_real.size(0)}")
                # 统计模型占用的系统内存（单位：MB）
                mem_bytes = sum(param.element_size() * param.nelement() for param in model.parameters())
                mem_mb = mem_bytes / (1024 ** 2)
                print(f"Model memory usage: {mem_mb:.2f} MB")
                outputs = model(data_batch_real)
                _, predicted = torch.max(outputs.data, 1)
                # 计算定位误差并收集所有误差
                distance_errors = []
                for true_label, pred_label in zip(label_int_batch.cpu().numpy(), predicted.cpu().numpy()):
                    label_total[true_label] += 1
                    label_pred_total[pred_label] += 1
                    if true_label == pred_label:
                        label_correct[true_label] += 1
                    x1, y1 = coords[true_label]['true_x'], coords[true_label]['true_y']
                    x2, y2 = coords[pred_label]['true_x'], coords[pred_label]['true_y']
                    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    total_distance_error += dist
                    unmatched_count += 1
                    distance_errors.append(dist)
                    
            if num_batches > 0:
                print(f"Average MSE : {total_mse / num_batches:.6f}")
            # 绘制CDF图
            if len(distance_errors) > 0:
                sorted_errors = np.sort(distance_errors)
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                error = pd.DataFrame({
                    '2finetune': sorted_errors,
                })
                cdf_csv_path = os.path.join(output_dir, '2finetune.csv')
                error.to_csv(cdf_csv_path, index=False)
                plt.figure()
                plt.plot(sorted_errors, cdf, marker='.', linestyle='-')
                plt.xlabel('Localization Error (meters)')
                plt.ylabel('CDF')
                plt.title('CDF of Localization Error')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("No localization errors to plot CDF.")
            total += label_int_batch.size(0)
            correct += (predicted == label_int_batch).sum().item()
        if unmatched_count > 0:
            avg_distance_error = total_distance_error / total
            print(f"MLE : {avg_distance_error:.4f}")
        else:
            print("All labels matched, no localization error for unmatched labels.")
        accuracy = correct / total
        recall = recall_score(label_int_batch.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        precision = precision_score(label_int_batch.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test precision: {precision:.4f}")
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