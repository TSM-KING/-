# 无监督
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 定义加载CIFAR-10的meta文件
def load_cifar10_meta(data_dir):
    """加载CIFAR-10的meta数据，包括类别名称"""
    with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    return meta[b'label_names']

# 定义加载CIFAR-10数据集的函数
def load_cifar10_batch(filename):
    """ 从一个二进制文件中加载CIFAR-10数据 """
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']   # 图像数据（每张图像为3072维的向量）
    labels = batch[b'labels']  # 图像标签
    return data, labels

def load_cifar10_data(data_dir):
    """ 加载所有训练和测试数据 """
    # 加载训练集数据（包括 5 个数据批次）
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(batch_file)
        train_data.append(data)
        train_labels.extend(labels)
    
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels)  # 确保train_labels是一个NumPy数组

    # 加载测试集数据
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    test_labels = np.array(test_labels)  # 确保test_labels是一个NumPy数组
    
    return train_data, train_labels, test_data, test_labels

# 假设数据集存储在当前目录的文件夹中
data_dir = 'cifar-10-batches-py'
# 加载数据
train_images, train_labels, test_images, test_labels = load_cifar10_data(data_dir)

# 数据预处理：
# 将图像数据归一化到 [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 将标签转换为一维数组
train_labels = train_labels.flatten()  # 现在train_labels是NumPy数组，可以调用flatten()
test_labels = test_labels.flatten()    # 同样，test_labels也是NumPy数组

# 加载类别名称
label_names = load_cifar10_meta(data_dir)

# 将训练数据划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# 将数据展平成一维向量（每张图像 3072 维）
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = test_images.reshape(test_images.shape[0], -1)

# 使用 K-Means 聚类，K=10 因为 CIFAR-10 有 10 个类别
kmeans = KMeans(n_clusters=10,n_init=10, random_state=42)
kmeans.fit(X_train_flat)

# 聚类结果
y_pred_train = kmeans.predict(X_train_flat)
y_pred_val = kmeans.predict(X_val_flat)
y_pred_test = kmeans.predict(X_test_flat)

# 评估 K-Means 聚类效果
# 计算纯度（Purity）指标：纯度衡量了每个簇中多数类别的比例
def purity_score(y_true, y_pred):
    """ 计算聚类的纯度 """
    contingency_matrix = np.zeros((10, 10), dtype=int)
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# 计算纯度
purity = purity_score(test_labels, y_pred_test)
print(f"Purity: {purity:.4f}")

# 使用混淆矩阵来可视化聚类结果
cm = confusion_matrix(test_labels, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for K-Means (CIFAR-10)')
plt.show()

# 将预测标签转换为类别名称
y_pred_labels = y_pred_test.astype(int)  # 确保预测标签是整数类型


# 创建 DataFrame 用于展示真实标签与预测标签
df = pd.DataFrame({
    'True Label': [label_names[label] for label in test_labels],
    'Predicted Label': [label_names[label] for label in y_pred_test]
})

# 打印分类结果的一部分（前10个样本）
print("\nClassification Results (True Label vs Predicted Label):")
print(df.head(10))  # 显示前10个预测结果
