from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
# t-sne的输入数据需要是（batch_size,features_dim）的形式
# 1. 读取数据
file_path = r"data\processedData\FLOOR3\all_data.csv"   
location_vector_csv = r"model\v1\output\location_vector_v2.csv"  

location_vector = pd.read_csv(location_vector_csv)
data = pd.read_csv(file_path)
sf=11
# 2. 筛选出 sf=sf 的数据
sf_data = data[data['sf']== sf]
# location_list=["point1","370","360","350","336","1m","328","308"]
# 读取 location_vector 数据
sf_data = sf_data[sf_data['location_id'].isin(location_vector['location_id'].values)]
sf_data['idx'] = sf_data['idx'].astype(int)
# sf_data = sf_data[sf_data['location_id'].isin(location_list)]

# features = sf_data[['average_rssi', 'snr','median_rssi','mode_rssi',"rssi_skewness","rssi_kurtosis"]].values
features = sf_data[['average_rssi','rssi_variance', 'median_rssi', 'mode_rssi', "snr","residual"]].values

tsne = TSNE(n_components=2, random_state=42,perplexity=30, max_iter=1000, learning_rate=200)
tsne_results = tsne.fit_transform(features)

# 将降维结果添加到数据框中
sf_data['tsne_1'] = tsne_results[:, 0]
sf_data['tsne_2'] = tsne_results[:, 1]


# 5. 设置颜色调色板
scientific_colors = [
    '#2E86C1',  # 深蓝（Pantone 7694 C）
    '#C0392B',  # 深红（Pantone 7621 C）
    '#1A5276',  # 普鲁士蓝
    '#922B21',  # 赤陶土色
    '#28B463',  # 翡翠绿
    '#D4AC0D',  # 琥珀金
    '#6C3483',  # 皇室紫
    '#BA4A00',  # 锈橙色
    '#148F77',  # 孔雀石绿
    '#7D3C98',  # 数码紫
    '#A04000',  # 深赭色
    '#1F618D',  # 钢蓝色
    '#CD6155',  # 陶土红
    '#52BE80',  # 薄荷绿
    '#AF601A',  # 皮革棕
    '#884EA0',  # 宝石紫
    '#C0392B',  # 重复色（需要时可扩展）
    '#1A5276'   # 重复色
]
# 4. 可视化
unique_labels = sf_data['idx'].unique()
n_labels = len(unique_labels)
palette = sns.color_palette(scientific_colors * (n_labels//len(scientific_colors)+1))[:n_labels]
# 1. 增大画布尺寸（核心！让图更宽）
plt.figure(figsize=(14, 10))  # 宽高比16:9，可改20:10等更大尺寸

# 2. scatterplot 保持原参数
sns.scatterplot(
    x='tsne_1', y='tsne_2', 
    hue='idx', 
    palette=palette, 
    data=sf_data, 
    legend="full",
    s=60  # 增大点的大小
)

# 优化图例点大小和整体布局
plt.grid(True)
plt.xlabel("t-SNE Dimension 1", fontsize=24, fontname='Arial')
plt.ylabel("t-SNE Dimension 2", fontsize=24, fontname='Arial')
plt.xticks(fontsize=20, fontname='Arial')
plt.yticks(fontsize=20, fontname='Arial')

# 关键修改：设置坐标轴比例为1:1，避免图形被拉伸或压缩
plt.gca().set_aspect('equal', adjustable='box')

# 调整图例字体和位置，使其居中且字体为Arial 20
# leg = plt.legend(
#     title='Location ID', 
#     bbox_to_anchor=(1.02, 0.5),  # 让图例在右侧垂直居中
#     loc='center left', 
#     prop={'size': 20, 'family': 'Arial'}
# )
leg = plt.legend(
    title='Location ID', 
    bbox_to_anchor=(1.02, 0.5),  
    loc='center left', 
    prop={'size': 20, 'family': 'Arial'},  # 图例每一项的字体
    title_fontproperties={'size': 20, 'family': 'Arial'}  # 标题“Location ID”的字体
)
if leg:
    for handle in leg.legend_handles:
        handle._sizes = [120]  # 图例点更大

# 调整布局参数，让散点图在画布中居中且更宽松
plt.subplots_adjust(
    left=0.1,   # 左侧留白
    right=0.8,  # 右侧给图例留空间，让散点图更居中
    bottom=0.1, # 底部留白
    top=0.9     # 顶部留白，避免散点图贴顶部
)
plt.tight_layout()

# 保存图片（含图例）
output_dir = r"data/processedData/FLOOR3"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(
    f"{output_dir}/residual_sf_{sf}.png", 
    dpi=300, 
    bbox_inches='tight'  # 保证保存时完整包含元素
)
plt.show()