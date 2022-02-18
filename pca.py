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
from scipy import stats
# 下载数据集
iris = datasets.load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names
# 一、数据预处理
# 重复和缺失数据
df = pd.DataFrame(x)
print("data中缺失数据情况",df.isnull().sum())
print("data中重复数据情况",df.duplicated().sum())

# 二、数据探索
# 1 数据描述
print("".format(df.info()))
print("数据总分布情况:\n{}".format(df.describe(include="all")))
print("数据总分布情况:\n{}".format(df.describe(include="all")))
# 1.1 直方图判断数据分布
columns = df.columns
print(df.head())
for column in columns:
    #print(df[column])
    sns.displot(x,x = df[column],kde= True)
    plt.title("column{}".format(column) )
plt.show()
# result: 根据直方分布图 预估分布：column 0 1 正态分布，23 不符合正态分布
# 统计column 后面两列的分布情况
print("第二列和第三列的出现次数\n")
print(len(df[2].unique()),len(df[3].unique()))
# 画出第二列和四散列
sns.swarmplot(data=df, x=df[2])
#plt.scatter(list(range(len(df))),df[2])
plt.title("the scatter of the second column ")
plt.show()
# 根据分布去除离群点
#比如根据df[1]的累积分布函数，
l = stats.norm.cdf(df[1],loc=3.057, scale=0.43)
print("2.8 在第二列出现的概率",l)
# 输出F(x)概率小于0.003，大于0.997索引
l = [True if (x<0.003 or x > 0.997) else False for i in l]
l = np.array(l)
l.reshape(1,-1)
df = df.loc[list(l)]
print("根据第二列的离群点情况，删除离群点后的长度",len(l))

# 1.2 fitter判断数据符合哪个分布
from fitter import Fitter
for column in columns:
    f = Fitter(df[column],distributions=['norm', 't', 'laplace']) #, distributions=['norm', 't', 'laplace'
    f.fit()
    print("第column列的分布",column)
    print(f.get_best(method='sumsquare_error'))
print("________________________")
# result:

# 1.3 检验数据是否某个分布
for column in columns:
    l = stats.kstest(df[column], "norm", (df[column ].mean(), df[column ].std()))
    print("第column列数据符合norm的kstest结果是",column)
    print(l)
# result : 前两列属于正态分布

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
# 评估kmeans 的聚类结果
