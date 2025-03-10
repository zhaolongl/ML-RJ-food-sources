import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # 使用 'Agg' 后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

# 读取Excel文件
file_path = '饲喂方式整理数据.xlsx'  # 请将 'path/to/your/file.xlsx' 替换为你的文件路径
data = pd.read_excel(file_path)

# 准备建模数据
X = data.iloc[:, 1:]  # 特征
y = data.iloc[:, 0]   # 目标变量

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义支持向量机分类器
svm_clf = SVC(random_state=42)

# 定义参数网格
param_grid = {
    'C': [0.0001, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.00001, 0.1, 0.01]
}

# 使用网格搜索进行超参数优化
grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数进行预测
best_svm_clf = grid_search.best_estimator_
y_pred_svm = best_svm_clf.predict(X_test)

# 评估模型
print("分类报告:")
print(classification_report(y_test, y_pred_svm))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred_svm))

# 可视化最佳超参数组合的混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")  # 修改为Blues颜色映射
plt.title("Confusion Matrix (Best Params)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('confusion_matrix_svm_best.pdf')  # 保存图像
plt.close()

# 使用更差的超参数组合
bad_svm_clf = SVC(C=0.00001, kernel='rbf', gamma=0.001, random_state=42)
bad_svm_clf.fit(X_train, y_train)
y_pred_bad_svm = bad_svm_clf.predict(X_test)

# 评估模型
print("分类报告 (Bad Params):")
print(classification_report(y_test, y_pred_bad_svm))
print("混淆矩阵 (Bad Params):")
print(confusion_matrix(y_test, y_pred_bad_svm))

# 可视化更差超参数组合的混淆矩阵
conf_matrix_bad = confusion_matrix(y_test, y_pred_bad_svm)
sns.heatmap(conf_matrix_bad, annot=True, fmt="d", cmap="Blues")  # 修改为Blues颜色映射
plt.title("Confusion Matrix (Bad Params)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig('confusion_matrix_svm_bad.pdf')  # 保存图像
plt.close()

# 可视化超参数的重要性
results = pd.DataFrame(grid_search.cv_results_)
scores = results.mean_test_score.values.reshape(len(param_grid['C']),
                                                len(param_grid['kernel']),
                                                len(param_grid['gamma']))

# 将 kernel 参数映射为数值
kernel_mapping = {'linear': 0, 'rbf': 1}
z_grid_values = [kernel_mapping[kernel] for kernel in param_grid['kernel']]
z_grid, x_grid, y_grid = np.meshgrid(z_grid_values, param_grid['C'], param_grid['gamma'], indexing='ij')

# 将得分展开为与网格相同的形状
scores_reshaped = scores.ravel()

# 绘制3D图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D图形
sc = ax.scatter(x_grid, y_grid, z_grid, c=scores_reshaped, cmap='coolwarm')  # 修改为coolwarm颜色映射
plt.colorbar(sc)
ax.set_xlabel('C')
ax.set_ylabel('Gamma')
ax.set_zlabel('Kernel')
plt.title('Grid Search Mean Test Scores in 3D (SVM)')
plt.savefig('grid_search_scores_3d_svm.pdf')  # 保存图像
plt.close()
