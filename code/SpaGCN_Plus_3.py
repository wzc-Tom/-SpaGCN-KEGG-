#name:SpaGCN_Plus_1.py
#env:syf_spa_py3.7
#安装必要的库
## need further
import sys
import os,csv,re
import pandas as pd ##
import numpy as np ##
import scanpy as sc ##
import math
import SpaGCN as spg ##
from scipy.sparse import issparse
import random, torch ##
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
#import zipfile 
#import gzip  
#import shutil 
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score  
# from sklearn.neighbors import NearestNeighbors 

#函数：生成文件路径 
def generate_file_paths(filenames, base_dir):  
    return [Path(base_dir) / filename for filename in filenames]  

#函数：确保目录存在
def ensure_directory_exists(directory):  
    if not Path(directory).exists():  
        Path(directory).mkdir(parents=True)  

#函数：解压文件
'''def unzip_file(zip_path, extract_to='.'):  
    # zip_path: zip文件的路径；extract_to: 解压到的目录，默认为当前目录  
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:  
        zip_ref.extractall(extract_to)  

def unzip_gz_file(gz_path, output_path):  
    with gzip.open(gz_path, 'rb') as f_in:  
        with open(output_path, 'wb') as f_out:  
            shutil.copyfileobj(f_in, f_out) 
'''
#数据来源：Brain
#[]中路径后面不要添加注释！
#配置变量区
GSE_name = 'BrainEpendymoma_GSE195661'
base_dir_1 = f"{GSE_name}/"
sample_id_part1 = '715'
sample_id_part2 = '459_2'
zimu = 'B'

#解压文件
#ensure_directory_exists(f"STdata/Download/Brain/{GSE_name}/Users/rf/Desktop/0222_clean/EPN_visium_012122/update/neuro_onc_spatial_files/spatial_{sample_id_part2}")
#zip_file_path = f"STdata/Download/Brain/{GSE_name}/Users/rf/Desktop/0222_clean/EPN_visium_012122/update/neuro_onc_spatial_files/{sample_id_part2}.zip"  
#extract_dir = f"STdata/Download/Brain/{GSE_name}/Users/rf/Desktop/0222_clean/EPN_visium_012122/update/neuro_onc_spatial_files/spatial_{sample_id_part2}"    
#unzip_file(zip_file_path, extract_dir)

'''if not os.path.exists(f"{GSE_name}/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}.tif"):
    gz_file_path = f"STdata/Download/Brain/{GSE_name}/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}.tif.gz" # .gz文件路径  
    output_file_path = f"STdata/Download/Brain/{GSE_name}/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}.tif"
    unzip_gz_file(gz_file_path, output_file_path)
'''
#路径构建
'''ensure_directory_exists(f"BrainEpendymoma_GSE195661/results/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}")
ensure_directory_exists(f"STdata/Sample_results/results_{sample_id_part2}")'''

filenames_1 = [
    f'GSE195661_RAW/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}_filtered_feature_bc_matrix.h5',
    f"GSE195661_spaceranger_spatial_dirs/{sample_id_part2}/spatial/tissue_positions_list.csv",  
    f"GSE195661_RAW/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}.tif"
    ]  
paths_1 = generate_file_paths(filenames_1, base_dir_1)  

base_dir_2 = f'{base_dir_1}/results/results_{sample_id_part2}'
ensure_directory_exists(base_dir_2)
filenames_2 = [
    'map.jpg',
    'results.h5ad',
    'pred.png',
    'meta_gene.png'
    ]
paths_2 = generate_file_paths(filenames_2, base_dir_2)

data_path = f'BrainEpendymoma_GSE195661/sample/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}/adj_map.csv'
ensure_directory_exists(f'BrainEpendymoma_GSE195661/sample/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}')
adata_path = paths_1[0]   
spatial_path = paths_1[1]  
image_path = paths_1[2] 
sample_path = f'BrainEpendymoma_GSE195661/sample/GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}/sample_data_GSM5844{sample_id_part1}_{zimu}1_{sample_id_part2}.h5ad'

map_path = paths_2[0]
results_path = paths_2[1]
pred_path = paths_2[2]
meta_gene_path =paths_2[3]

#原始数据读入
from scanpy import read_10x_h5
adata = read_10x_h5(adata_path)# 读取10x Genomics的h5格式单细胞测序数据，存入adata对象
spatial = pd.read_csv(spatial_path,sep=",",header=None,na_filter=False,index_col=0)# 读取空间信息CSV文件，不包含列名（header=None），索引设置为第一列（index_col=0），不自动过滤缺失值（na_filter=False）
#修改adata###(为什么不adata.obs["x_array"]=spatial[2])？？？
adata.obs["x1"]=spatial[1]
adata.obs["x2"]=spatial[2]
adata.obs["x3"]=spatial[3]
adata.obs["x4"]=spatial[4]
adata.obs["x5"]=spatial[5]
adata.obs["x_array"]=adata.obs["x2"]
adata.obs["y_array"]=adata.obs["x3"]
adata.obs["x_pixel"]=adata.obs["x4"]
adata.obs["y_pixel"]=adata.obs["x5"]
#保留被捕获的数据
adata=adata[adata.obs["x1"]==1]
#数据格式处理
adata.var_names=[i.upper() for i in list(adata.var_names)]
adata.var["genename"]=adata.var.index.astype("str")
#保存处理后的adata，包括.X、obs.[单细胞测序数据、空间信息CSV文件（x1-5,x_,y_）].x1==1、.var、.uns。
adata.write_h5ad(sample_path)

#scpy读取AnnData实例adata(gene expression and spatial location)
adata=sc.read(sample_path)
#读取组织学图像信息
img=cv2.imread(image_path)

#从adata中提取空间、像素坐标信息，并将这些信息转换为列表
#将单细胞测序数据与组织学图像进行空间对齐
x_array=adata.obs["x_array"].tolist()
y_array=adata.obs["y_array"].tolist()
x_pixel=adata.obs["x_pixel"].tolist()
y_pixel=adata.obs["y_pixel"].tolist()

#可视化地检查这些坐标是否正确地映射到了图像上的相应位置
img_new=img.copy()
for i in range(len(x_pixel)):
    x=x_pixel[i]
    y=y_pixel[i]
    img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0
    #在img_new图像的对应坐标位置绘制一个40x40像素的黑色方块，方块的中心位于(x, y)
cv2.imwrite(map_path, img_new)

#计算邻接矩阵
s=2### 组织学权重，可能需要修改；原为1
b=49#提取颜色强度时每个点的面积
adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
#组织学信息不可用时备选方案
#adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
#这里原代码有误
np.savetxt(data_path, adj, delimiter=',')# 使用逗号作为分隔符，即保存为CSV格式

#读取数据后预处理
adata=sc.read(sample_path)### 原代码如此，疑似重复冗余加载
adj=np.loadtxt(data_path, delimiter=',')
adata.var_names_make_unique()
spg.prefilter_genes(adata,min_cells=3) # 只分析相对高表达基因 ###可能需要调整
spg.prefilter_specialgenes(adata)
#对UMI（Unique Molecular Identifiers，唯一分子标识符）计数进行标准化处理，并对标准化后的结果取对数转换
#预处理模块pp
sc.pp.normalize_per_cell(adata)#normalize_per_cell函数对每个细胞进行标准化，将每个基因的UMI计数除以该细胞的总UMI计数（或乘以一个缩放因子，如10,000）
sc.pp.log1p(adata)#log1p函数对标准化后的表达量取对数（log1p是log(1+x)的缩写），使得数据的分布更加接近正态分布，（是自然对数）

#确定超参数p
p=0.5 ### 邻域对总表达量的贡献百分比，根据情况需要修改 
l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)# p由l控制
#l用于调整某种与图结构相关的计算，比如邻域聚合的权重或范围，从而间接控制p。
#通过搜索不同的l值，可以找到使p达到目标值或使聚类/特征提取性能最优的l。
#也因此，spg.search_l函数需要邻接矩阵adj作为输入，以便在搜索过程中考虑图的结构信息。

'''
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  
from sklearn.metrics import pairwise_distances #单独列出
# 定义设置随机种子的函数
def set_seed(r_seed, t_seed, n_seed):
    np.random.seed(n_seed)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(t_seed)
        torch.cuda.manual_seed_all(t_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dunn_index_score(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    intra_cluster_dists = []
    inter_cluster_dists = []

    for label in unique_labels:
        cluster_points = X[labels == label]
        intra_cluster_dists.append(np.max(pairwise_distances(cluster_points)))

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_points_i = X[labels == unique_labels[i]]
            cluster_points_j = X[labels == unique_labels[j]]
            inter_cluster_dists.append(np.min(pairwise_distances(cluster_points_i, cluster_points_j)))

    dunn = np.min(inter_cluster_dists) / np.max(intra_cluster_dists)
    return dunn

# 初始化一个字典来存储每个聚类数及其对应的五个评估指标的总分数  
cluster_scores = {}  

# 设置种子
r_seed = 100
t_seed = 100
n_seed = 100
set_seed(r_seed, t_seed, n_seed)

best_n_clusters = 0
best_sil_score = -1    
best_chi_score = -1  
best_dbi_score = float('inf')

# 尝试不同的聚类数
out_list = []
for n_clusters in range(2, 20):
    set_seed(r_seed, t_seed, n_seed)  # 在每次迭代中设置种子以确保一致性
    print(f"尝试聚类数: {n_clusters}")
    res = spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    print(f"找到的分辨率: {res}")
    set_seed(r_seed, t_seed, n_seed)
    clf=spg.SpaGCN()
    clf.set_l(l)
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs['pred'] = y_pred
    # 计算轮廓系数  
    sil_score = silhouette_score(adata.X, adata.obs['pred'].to_numpy())  
    # 计算Calinski-Harabasz Index  
    chi_score = calinski_harabasz_score(adata.X.toarray(), adata.obs['pred'].to_numpy())  
    # 计算Davies-Bouldin Index  
    dbi_score = davies_bouldin_score(adata.X.toarray(), adata.obs['pred'].to_numpy())  
    # 计算dunn Index
    dunn_score = dunn_index_score(adata.X.toarray(), adata.obs['pred'].to_numpy())
    # 计算总分数
    total_score = sil_score + chi_score + dunn_score + (1 / (dbi_score + 1e-10))  # 加1e-10防止除以0
    # 更新cluster_scores字典  
    if n_clusters not in cluster_scores:  
       cluster_scores[n_clusters] = total_score  
    else:  
       cluster_scores[n_clusters] = max(cluster_scores[n_clusters], total_score)  
    
    out_list.append(f"聚类数: {n_clusters}, 轮廓系数: {sil_score}, CHI: {chi_score}, DBI: {dbi_score}，Dunn Index: {dunn_score},总分数：{total_score}")  
        
# 找到总分数最高的聚类数  
best_n_clusters = max(cluster_scores, key=cluster_scores.get)
  
print(out_list)  
print(f'最佳聚类数: {best_n_clusters}')  
'''

# 定义设置随机种子的函数
def set_seed(r_seed, t_seed, n_seed):
    np.random.seed(n_seed)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    if torch.cuda.is_available():# 检查CUDA设备可及性
        torch.cuda.manual_seed(t_seed)
        torch.cuda.manual_seed_all(t_seed)
        torch.backends.cudnn.deterministic = True# 打开确定性模式，确保可重复性
        torch.backends.cudnn.benchmark = False# 关闭benchmark模式，提高可重复性

# 计算聚类结果的 Dunn Index
def dunn_index_score(X, labels):  
    # 计算 Dunn Index  
    n_clusters = len(set(labels))# labels在函数find_best_k()中被定义
    if n_clusters <= 1:  
        return float('inf')  # 当只有一个聚类时，Dunn Index 无意义，故返回无穷大
      
    # 计算类内最大距离  
    max_diam = 0# 初始化为0，存储类内最大距离
    for i in range(n_clusters):  
        cluster_points = X[labels == i]# 生成一个布尔数组，用作X的索引，以提取聚类点
        if len(cluster_points) > 1:  # 检查每个聚类是否包含多于一个点
            cluster_diam = max([np.linalg.norm(cluster_points[j] - cluster_points[k]) for j in range(len(cluster_points)) for k in range(j + 1, len(cluster_points))]) # 计算欧几里得距离并找出其最大值 
            # 这里用了二重循环，大的数据集可能需要修改
            '''
            pairwise_distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :], axis=2)
            cluster_diam = np.max(pairwise_distances)  # Find the maximum distance
            '''
            ### 一种可行的不错的修改方案
            if cluster_diam > max_diam:  
                max_diam = cluster_diam  
      
    # 计算类间最小距离  
    min_dist = float('inf')# 初始为无穷大
    '''
    nn = NearestNeighbors(n_neighbors=2)# 创建一个 NearestNeighbors 实例 ##n_neighbors设置为多少值得权衡
    nn.fit(X)# 拟合数据（“学习”X），将距离信息存储在NearestNeighbors实例的内部数据结构
    distances, indices = nn.kneighbors(X)  
    '''
    #注释掉是因为在我修改后的逻辑中这显得多余了，不过仍然可以有借鉴价值
    for i in range(len(X)):  
        for j in range(i + 1, len(X)):  
            if labels[i] != labels[j]:
                dist = np.linalg.norm(X[i] - X[j])
                if dist < min_dist:  
                   min_dist = dist  
      
    return min_dist / max_diam  

def find_best_k(features, max_k=15):
    sse = [] # 随着k的增加，SSE通常会减小，因为更多的聚类中心可以更好地适应数据点
    silhouette_scores = [] # 计算每个点到其同类中其他点的平均距离（紧致性）以及到最近的不同类中的点的平均距离（分离性）
    calinski_harabasz_scores = []  
    davies_bouldin_scores = []  #越小越好
    dunn_scores = []  
    for k in range(4, max_k + 1):#2，3附近有局部收敛最优现象，予以屏蔽
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(features.toarray(), labels))  
        davies_bouldin_scores.append(davies_bouldin_score(features.toarray(), labels))  
        dunn_scores.append(dunn_index_score(features.toarray(), labels)) 
     
    plt.figure(figsize=(10, 5))
    plt.plot(range(4, max_k + 1), sse, marker='o')
    plt.title('elbow-method-K')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.savefig(os.path.join(base_dir_2, 'elbow_method.png'))  # 保存肘部法图像
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(4, max_k + 1), silhouette_scores, marker='o')
    plt.title('silhouette-K')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(base_dir_2, 'silhouette_score.png'))  # 保存轮廓系数图像，下略
    plt.close()
    #The silhouette score tends to favor fewer clusters.
    #A low k can maximize the silhouette score due to increased separation between two large clusters, even when finer clustering is required.
    plt.figure(figsize=(10, 5))  
    plt.plot(range(4, max_k + 1), calinski_harabasz_scores, marker='o')  
    plt.title('Calinski-Harabasz Index-K')  
    plt.xlabel('Number of clusters')  
    plt.ylabel('Calinski-Harabasz Score')  
    plt.savefig(os.path.join(base_dir_2, 'calinski_harabasz_index.png'))  
    plt.close()  
  
    plt.figure(figsize=(10, 5))  
    plt.plot(range(4, max_k + 1), davies_bouldin_scores, marker='o')  
    plt.title('Davies-Bouldin Index-K')  
    plt.xlabel('Number of clusters')  
    plt.ylabel('Davies-Bouldin Score')  
    plt.savefig(os.path.join(base_dir_2, 'davies_bouldin_index.png'))  
    plt.close()  
  
    plt.figure(figsize=(10, 5))  
    plt.plot(range(4, max_k + 1), dunn_scores, marker='o')  
    plt.title('Dunn-Index-K')  
    plt.xlabel('Number of clusters')  
    plt.ylabel('Dunn Score')  
    plt.savefig(os.path.join(base_dir_2, 'dunn_index.png'))  
    plt.close() 
    print("Please check the generated plots to determine the best k value.") 
    sys.stdout.flush()
    best_k = int(input('k_cluster='))
    return best_k

best_k = find_best_k(adata.X) 

n_clusters = best_k
# 设置随机种子，这里使用100
r_seed = 100
t_seed = 100
n_seed = 100
set_seed(r_seed, t_seed, n_seed)
from scipy.sparse import issparse

if issparse(adata.X):
    adata.X = adata.X.toarray()
# 已知聚类数，利用search_res搜索合适的分辨率
res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
### 参数来源于SpaGCN样例代码，可能需要修改；如学习率
set_seed(r_seed, t_seed, n_seed)
clf=spg.SpaGCN()# 创建SpaGCN实例
clf.set_l(l)# 设置超参数
# 运行SpaGCN，训练模型
clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
### 参数来源于SpaGCN样例代码，可能需要修改；200我担心过拟合
# 预测、保存结果
y_pred, prob=clf.predict()
adata.obs["pred"]= y_pred
adata.obs["pred"]=adata.obs["pred"].astype('category')### 转换为分类标签的意义是什么？

# 细化聚类结果，在原代码中，这一步是可选的
###shape参数设置为"hexagon"或"square"，这取决于数据的类型（Visium数据使用"hexagon"，ST数据使用"square"）
# 不考虑组织学信息的计算邻接矩阵
adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
# 聚类细化和保存
refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
adata.obs["refined_pred"]=refined_pred
adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
# 保存结果
adata.write_h5ad(results_path)

adata=sc.read(results_path)### 111这么多adata会有影响吗
#adata.obs should contain two columns for x_pixel and y_pixel
#定义一个颜色列表
plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
#绘制预测结果的空间分布
domains="pred"
#确定细胞类型的数量，domains列的唯一值
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(pred_path, dpi=600)
plt.close()

#绘制精细化domain图
domains="refined_pred"
num_celltype=len(adata.obs[domains].unique())
adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(pred_path, dpi=600)
plt.close()

#Read in raw data
raw=sc.read(sample_path)
raw.var_names_make_unique()
raw.obs["pred"]=adata.obs["pred"].astype('category')
raw.obs["x_array"]=raw.obs["x2"]
raw.obs["y_array"]=raw.obs["x3"]
raw.obs["x_pixel"]=raw.obs["x4"]
raw.obs["y_pixel"]=raw.obs["x5"]
raw.X=(raw.X.toarray() if issparse(raw.X) else raw.X)#转换稀疏矩阵#Convert sparse matrix to non-sparse
#raw.raw=raw #原注释如此，不知其意
sc.pp.log1p(raw)

for i in range(0,int(n_clusters)):
    target = i# 识别SVGs的目标域
    #设置过滤标准
    min_in_group_fraction=0.8#Minium in-group expression fraction.
    min_in_out_group_ratio=1#Miniumn (in-group expression fraction) / (out-group expression fraction).
    min_fold_change=1.5#Miniumn (in-group expression) / (out-group expression).
    adj_2d=spg.calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    start, end= np.quantile(adj_2d[adj_2d!=0],q=0.001), np.quantile(adj_2d[adj_2d!=0],q=0.1)
    #使用spg.search_radius函数搜索一个合适的半径，使得目标簇中的每个点平均有10个邻近点
    r=spg.search_radius(target_cluster=target, cell_id=adata.obs.index.tolist(), x=x_array, y=y_array, pred=adata.obs["pred"].tolist(), start=start, end=end, num_min=10, num_max=14,  max_run=100)
    #检测相邻域
    nbr_domians=spg.find_neighbor_clusters(target_cluster=target,
                                       cell_id=raw.obs.index.tolist(), 
                                       x=raw.obs["x_array"].tolist(), 
                                       y=raw.obs["y_array"].tolist(), 
                                       pred=raw.obs["pred"].tolist(),
                                       radius=r,
                                       ratio=1/2)

    nbr_domians=nbr_domians[0:3]#只保留前3个邻近域进行后续分析
    de_genes_info=spg.rank_genes_groups(input_adata=raw,
                                    target_cluster=target,
                                    nbr_list=nbr_domians, 
                                    label_col="pred", 
                                    adj_nbr=True, 
                                    log=True)
    #从差异表达基因DEGs中筛选出空间可变基因SVGs
    de_genes_info=de_genes_info[(de_genes_info["pvals_adj"]<0.05)]
    filtered_info=de_genes_info
    filtered_info=filtered_info[(filtered_info["pvals_adj"]<0.05) &
                                (filtered_info["in_out_group_ratio"]>min_in_out_group_ratio) &
                                (filtered_info["in_group_fraction"]>min_in_group_fraction) &
                                (filtered_info["fold_change"]>min_fold_change)]
    filtered_info=filtered_info.sort_values(by="in_group_fraction", ascending=False)
    filtered_info["target_dmain"]=target
    filtered_info["neighbors"]=str(nbr_domians)
    print("SVGs for domain ", str(target),":", filtered_info["genes"].tolist())

    #Plot refinedspatial domains
    color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
    for g in filtered_info["genes"].tolist():
        raw.obs["exp"]=raw.X[:,raw.var.index==g]
        ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=g,color_map=color_self,show=False,size=100000/raw.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        tempname = f"{base_dir_2}/domain{target}_{g}.png"  
        plt.savefig(tempname, dpi=600)
        plt.close()
    
    #元基因寻找
    meta_name, meta_exp=spg.find_meta_gene(input_adata=raw,
                        pred=raw.obs["pred"].tolist(),
                        target_domain=target,
                        start_gene="GFAP",###起始基因设置
                        mean_diff=0,
                        early_stop=True,
                        max_iter=3,
                        use_raw=False)

    raw.obs["meta"]=meta_exp

    
    g="GFAP"
    raw.obs["exp"]=raw.X[:,raw.var.index==g]
    ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=g,color_map=color_self,show=False,size=100000/raw.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    tempname = f"{base_dir_2}/domain{target}_{g}.png"
    plt.savefig(tempname, dpi=600)
    plt.close()
 
    raw.obs["exp"]=raw.obs["meta"]
    ax=sc.pl.scatter(raw,alpha=1,x="y_pixel",y="x_pixel",color="exp",title=meta_name,color_map=color_self,show=False,size=100000/raw.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(meta_gene_path, dpi=600)
    plt.close()
