import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 使用 'Agg' 后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# 读取Excel文件
file_path = '花期数据.xlsx'  # 请将 'path/to/your/file.xlsx' 替换为你的文件路径
data = pd.read_excel(file_path)

# 确保所有列名都是字符串类型
data.columns = data.columns.astype(str)

# 准备建模数据
X = data.iloc[:, 1:]  # 特征
y = data.iloc[:, 0]   # 目标变量

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义随机森林分类器
clf = RandomForestClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [1, 100, 200],
    'max_depth': [1, 20, 30],
    'min_samples_split': [2, 5, 10],  # 注意，min_samples_split 不能为 1
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数进行预测
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

# 评估模型
print("分类报告:")
print(classification_report(y_test, y_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")  # 修改为Blues颜色映射
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('confusion_matrix_随机森林花期.pdf')  # 保存图像
plt.close()

# 可视化超参数的重要性
results = pd.DataFrame(grid_search.cv_results_)
scores = results.mean_test_score.values.reshape(len(param_grid['n_estimators']),
                                                len(param_grid['max_depth']),
                                                len(param_grid['min_samples_split']),
                                                len(param_grid['min_samples_leaf']),
                                                len(param_grid['max_features']))

# 3D 可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 设置x, y, z轴的数据
x = param_grid['n_estimators']
y = param_grid['max_depth']
z = param_grid['min_samples_split']
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

# 将得分展开为与网格相同的形状
scores_reshaped = scores.max(axis=(3, 4)).ravel()

# 绘制3D图形
sc = ax.scatter(x_grid, y_grid, z_grid, c=scores_reshaped, cmap='coolwarm')  # 修改为coolwarm颜色映射
plt.colorbar(sc)
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Max Depth')
ax.set_zlabel('Min Samples Split')
plt.title('Grid Search Mean Test Scores in 3D')
plt.savefig('grid_search_scores_3d_随机森林花期.pdf')  # 保存图像
plt.close()
