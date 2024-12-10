import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

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

# 标准化数据（对于 XGBoost，标准化通常不是必要的，但可以尝试）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # 展平每张图片
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1))  # 同样对验证集进行标准化
X_test = scaler.transform(test_images.reshape(test_images.shape[0], -1))  # 对测试集进行标准化

# 降维（使用 PCA）
pca = PCA(n_components=0.95)  # 保留 95% 的方差
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# 使用 XGBoost 训练模型
dtrain = xgb.DMatrix(X_train_pca, label=y_train)
dval = xgb.DMatrix(X_val_pca, label=y_val)
dtest = xgb.DMatrix(X_test_pca, label=test_labels)

# 设置 XGBoost 参数
params = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 10,               # CIFAR-10 有 10 类
    'eval_metric': 'merror',       # 错误率（分类错误的比例）
    'max_depth': 6,                # 树的最大深度
    'eta': 0.1,                    # 学习率
    'subsample': 0.8,              # 随机采样比例
    'colsample_bytree': 0.8        # 树的列采样比例
}

# 超参数调优（使用 GridSearchCV）
param_grid = {
    'max_depth': [4, 6, 8],
    'eta': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}


# 这里使用 xgb.train 手动优化
evals = [(dtrain, 'train'), (dval, 'eval')]
evals_result = {}
bst = xgb.train(params, dtrain, num_boost_round=20, evals=evals, early_stopping_rounds=10, evals_result=evals_result, verbose_eval=True)

# 预测测试集
y_pred = bst.predict(dtest)

# 计算评估指标
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, average='weighted')
recall = recall_score(test_labels, y_pred, average='weighted')
f1 = f1_score(test_labels, y_pred, average='weighted')

# 打印评估结果
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# 绘制准确率曲线
train_accuracies = evals_result['train']['merror']
val_accuracies = evals_result['eval']['merror']
train_accuracy = [1 - err for err in train_accuracies]
val_accuracy = [1 - err for err in val_accuracies]

plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.xlabel('Boosting Round')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for XGBoost (CIFAR-10)')
plt.show()

# 将预测结果转换为整数类型（这通常是需要的，因为y_pred可能是浮动型数据）
y_pred_int = y_pred.astype(int)

# 创建 DataFrame 用于展示真实标签与预测标签
df = pd.DataFrame({
    'True Label': [label_names[label] for label in test_labels],
    'Predicted Label': [label_names[label] for label in y_pred_int]  # 这里确保预测标签是整数类型
})

# 打印分类结果的一部分（例如前10个样本）
print("\nClassification Results (True Label vs Predicted Label):")
print(df.head(10))  # 显示前10个预测结果



