import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

# 读取数据
path_openfile_name = './data/150-HighFre0511.xlsx'
data = pd.read_excel(path_openfile_name)

# 数据预处理
X = data.iloc[:, 2:-3].values  # 假设特征从第三列开始到倒数第三列
y = data.iloc[:, -3:].values  # 输出数据为最后三列

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# 定义高斯过程核
kernel = C(1.0, (1e-4, 1e2)) * RBF(10, (1e-4, 1e2))

# 实例化高斯过程回归模型，并使用MultiOutputRegressor进行包装
gpr = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True))

# 训练模型
gpr.fit(X_train, y_train)

# 预测测试集
y_pred = np.column_stack([estimator.predict(X_test) for estimator in gpr.estimators_])

# 反标准化预测结果
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)

# 绘制预测结果与真实值对比图
plt.figure(figsize=(15, 5))
for i in range(y_test_real.shape[1]):  # 假设有三个输出
    plt.subplot(1, 3, i + 1)
    plt.plot(y_test_real[:, i], 'o-', label='Real')
    plt.plot(y_pred_real[:, i], '.-', label='Predicted')
    plt.title(f'Output {i+1}')
    plt.legend()
plt.tight_layout()
plt.show()
