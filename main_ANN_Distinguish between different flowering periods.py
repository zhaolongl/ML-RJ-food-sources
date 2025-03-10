import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用 'Agg' 后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# 读取Excel文件
file_path = '花期数据.xlsx'  # 请将 'path/to/your/file.xlsx' 替换为你的文件路径
data = pd.read_excel(file_path)

# 准备建模数据
X = data.iloc[:, 1:]  # 特征
y = data.iloc[:, 0]   # 目标变量

# 数据预处理
# 将目标变量进行编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 将数据转换为张量
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 定义全连接神经网络模型
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(int(input_size), int(hidden_size))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size), int(hidden_size))
        self.fc3 = nn.Linear(int(hidden_size), int(num_classes))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 定义参数网格
param_grid = {
    'hidden_size': [32, 64, 128],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64]
}

# 定义模型参数
input_size = X.shape[1]
num_classes = len(le.classes_)
num_epochs = 50

# 初始化结果存储
results = []

# 超参数搜索
for params in ParameterGrid(param_grid):
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # 初始化模型、损失函数和优化器
    model = NeuralNet(input_size, params['hidden_size'], num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 训练模型
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 在测试集上进行预测
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    # 评估模型
    accuracy = accuracy_score(y_true, y_pred)
    results.append((params['hidden_size'], params['learning_rate'], params['batch_size'], accuracy))

# 转换结果为DataFrame
results_df = pd.DataFrame(results, columns=['hidden_size', 'learning_rate', 'batch_size', 'accuracy'])

# 3D 可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置x, y, z轴的数据
x = results_df['hidden_size']
y = results_df['learning_rate']
z = results_df['batch_size']
c = results_df['accuracy']

# 绘制3D图形
sc = ax.scatter(x, y, z, c=c, cmap='coolwarm')
plt.colorbar(sc)
ax.set_xlabel('Hidden Size')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Batch Size')
plt.title('Grid Search Mean Test Scores in 3D (Neural Network)')
plt.savefig('grid_search_scores_3d_nn.pdf')  # 保存图像
plt.close()

# 可视化混淆矩阵
best_params = results_df.loc[results_df['accuracy'].idxmax()]
print("Best Parameters:", best_params)
best_model = NeuralNet(input_size, int(best_params['hidden_size']), num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=int(best_params['batch_size']), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=int(best_params['batch_size']), shuffle=False)

# 训练最佳模型
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        outputs = best_model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 在测试集上进行预测
best_model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for features, labels in test_loader:
        outputs = best_model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# 混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Best Params)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('confusion_matrix_nn_best.pdf')
plt.close()
