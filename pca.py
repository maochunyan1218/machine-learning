from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

# 下载数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names
# 一、数据预处理
# 重复和缺失数据
df = pd.DataFrame(x)
print("data中缺失数据情况".format(df.isnull().sum()))
print("data中重复数据情况".format(df.duplicated().sum()))
# 二、数据探索
# 1 数据描述
print("".format(df.info()))
print("数据总分布情况:\n{}".format(df.describe(include="all")))
# 数据分布
columns = df.columns
print(df.head())


for column in columns:
    #print(df[column])
    sns.displot(x,x = df[column],kde= True)
    plt.title("column{}".format(column) )
plt.show()
# 0、离群点检测，获取正确的样本
print("去除离群点之前样本大小：{}".format(len(x)))
estimator = LocalOutlierFactor(n_neighbors= 5).fit_predict(list(x))
# 取出正样本
estimator = [bool(i) for i in estimator]
x = x[estimator]
y = y[estimator]
print("去除离群点后样本大小：{}".format(len(x)))

# 一、根据PCA映射的二维图形状，选择合适的聚类算法及聚类数目
reduced = PCA(n_components = 2).fit_transform(x)
# 画出二维数据图
plt.subplot(221)
plt.scatter(reduced[:,0],reduced[:,1])
plt.title("feature image")
# result:型状可用kmeans，cluster选择2或者3

# 使用Kmeans
# 这里用原来的数据聚类，还是用降维后的数据聚类？
# 二、kmeans的评估：1、elbow 2、轮廓系数 3、
# 1、elbow
len(x)
inertias = []
for cluster in range(2,30):
    kmeans = KMeans(n_clusters= cluster).fit(reduced)
    inertias.append(kmeans.inertia_)
plt.subplot(222)
plt.plot(list(range(2,30)),inertias)
plt.scatter(list(range(2,30)),inertias)
#plt.title("different cluster num")
plt.title("elbow image")
# result:

# 2、轮廓系数
silhouette=[]
for cluster in range(2,30):
    kmeans = KMeans(n_clusters=cluster).fit(reduced)
    t = silhouette_score(x,kmeans.labels_)
    silhouette.append(t)
plt.subplot(223)
plt.plot(range(2,30),silhouette)
plt.scatter(range(2,30),silhouette)

#plt.title("different cluster num")
plt.xlabel("silhouette image")
#result:

# 3、cal_score
ch_score = []
for cluster in range(2,30):
    kmeans = KMeans(n_clusters=cluster).fit(reduced)
    t = calinski_harabasz_score(x,kmeans.labels_)
    ch_score.append(t)
plt.subplot(224)
plt.plot(list(range(2,30)),ch_score)
plt.scatter(list(range(2,30)),ch_score)
#plt.xlabel("different cluster num")
plt.xlabel("calinski_harabasz_score  image")
# result：
#plt.show()

# 确认cluster，得出聚类结果
kmeans = KMeans(n_clusters=3).fit(reduced)
print(kmeans.labels_)
