from sklearn.multioutput import MultiOutputRegressor
import pandas  as  pd
# 读取数据
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
"""   =======    获取数据 ======= """


path_openfile_name = './data/150-HighFre.xlsx'
input_table = pd.read_excel(path_openfile_name) #读取表格
# print(input_table)
input_table_rows = input_table.shape[0]  # 行数
input_table_colunms = input_table.shape[1]  # 列数
input_table_header = input_table.columns.values.tolist()  # 表头

print(" input_table_rows \n", input_table_rows)
print(" input_table_colunms \n", input_table_colunms)
print(" input_table_header \n", input_table_header)

print(" input_table \n", input_table )
print(" input_table  \n", input_table.shape)

xy=np.array(input_table)

x_data = xy[:, 2:-3]  #第三列到倒数第二列
y_data = xy[:, -3:]   #倒数第三列到最后一列
P_array = np.array(x_data)
C_array = np.array(y_data)

Paras_all = P_array  # N 个尺寸的结果 (N,4)
Curves_all = C_array  # N 个曲线的结果  (N,3,1)

# # """   =======    可以得到所有的数据 ======= """
print(" # 得到所有数据处理过后 N  个尺寸的结果形状\n", Paras_all.shape)
# print(" # 得到所有数据处理过后 N11  个尺寸的结果形状\n", Paras_all[0].shape)
print(" #得到所有数据处理过后  N  个曲线的结果形状\n", Curves_all.shape)
# print(" # N  个尺寸的结果", Paras_all)
print(" #得到所有数据处理过后 个尺寸\n", Paras_all[0])

print(" #得到所有数据处理过后 个尺寸\n", Paras_all[0][2])

"""   =======   多输入多输出回归 ======= """
X=Paras_all
y=Curves_all
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=4)
# 准备参数
other_params = {
    'eta': 0.1,             # 学习率
    'reg_alpha': 0.01,      # L1 正则化参数
    'reg_lambda': 0.01,     # L2 正则化参数
    'max_depth': 6          # 树的最大深度
}

# 训练模型
#创建了一个使用XGBoost的多输出回归模型，然后使用XGBRegressor作为基础估计器来拟合数据。
#objective='reg:squarederror' 代替了 'reg:linear',意味着使用平方误差作为损失函数。
multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(train_X,
                                                                                                                train_y)
check = multioutputregressor.predict(test_X)

ypred= multioutputregressor.predict(test_X)
print(ypred.shape)
print('MSE of prediction on boston dataset:111111111111111111', mean_squared_error(test_y, ypred))
print('\n')

print('test_y\n', test_y[-10:-1])# 取最后10个数据
print('ypred\n', ypred[-10:-1])

print('test_y.shape\n', test_y.shape)
print('ypred.shape\n', ypred.shape)

"""   =======    0511预测所有测试集数据 ======= """
"""   =======    画图 ======= """
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure()
plt.plot(test_y[:, -3:-2].flatten(), marker='o', label='y_Real')   #flatten()将多维数组降为一维
plt.plot(ypred[:, -3:-2].flatten(), marker='^', label="y_Predict")
plt.title("归一化相速 y_Real & y_Predict ")
plt.legend()
plt.show()
# [-2:-1] 对比图
plt.figure()
plt.plot(test_y[:,-2:-1].flatten(), marker='o', label='y_Real')
plt.plot(ypred[:,-2:-1].flatten(), marker='^', label="y_Predict")
plt.title("耦合阻抗 y_Real & y_Predict ")
plt.legend()
plt.show()
# [-1]  对比图
plt.figure()
plt.plot(test_y[:,-1].flatten(), marker='o', label='y_Real')
plt.plot(ypred[:,-1].flatten(), marker='^', label="y_Predict")
plt.title("衰减常数 y_Real & y_Predict ")
plt.legend()
plt.show()

# 对比分析
print('test_y\n',test_y[-10:-1,-3:-2],test_y[-10:-1,-2:-1],test_y[-10:-1,-1])
print('ypred\n',ypred[-10:-1,-3:-2],ypred[-10:-1,-2:-1],ypred[-10:-1,-1])

print('test_y[-10:-1,-3:-2] \n',test_y[-10:-1,-3:-2])
print('ypred[-10:-1,-3:-2] \n',ypred[-10:-1,-3:-2])

print('test_y[-10:-1,-2:-1] \n',test_y[-10:-1,-1])
print('ypred[-10:-1,-2:-1] \n',ypred[-10:-1,-1])

print('test_y[-10:-1,-1] \n',test_y[-10:-1,-1])
print('ypred[-10:-1,-1] \n',ypred[-10:-1,-1])


"""   =========== ==== ==== ==== ====     预测New 1218======= """
# 预测训练
pathpredict = "./data/1218HighFre20-1.xlsx"


# path_openfile_name = 'D:/CodeTF/GPR/data/HighFre/150-HighFre.xlsx'
input_table = pd.read_excel(pathpredict) #读取表格
# print(input_table)
input_table_rows = input_table.shape[0]  # 行数
input_table_colunms = input_table.shape[1]  # 列数
input_table_header = input_table.columns.values.tolist()  # 表头

print(" input_table_rows \n", input_table_rows)
print(" input_table_colunms \n", input_table_colunms)
print(" input_table_header \n", input_table_header)

print(" input_table \n", input_table)
print(" input_table  \n", input_table.shape)



xy=np.array(input_table)

x_datapredict = xy[:, 2:-3]
y_datapredict = xy[:, -3:]


P_arraypredict = np.array(x_datapredict)
C_arraypredict = np.array(y_datapredict)
# C_arraypredict = C_arraypredict.reshape(xypredict.shape[0],1)   #len=xy.shape[0] 数组长度 750

Paras_allpredict = P_arraypredict     #   N  个尺寸的结果 (N,4)
Curves_allpredict = C_arraypredict    # N 个曲线的结果  (N,3,1)


ypred_allpredict = multioutputregressor.predict(Paras_allpredict)
print(ypred.shape)
print('MSE of prediction on boston dataset:', mean_squared_error(Curves_allpredict, ypred_allpredict))

print('test_y \n', Curves_allpredict)
print('ypred \n', ypred_allpredict)

print('test_y \n', Curves_allpredict.shape)
print('ypred \n', ypred_allpredict.shape)

test_y=Curves_allpredict
ypred=ypred_allpredict

# [-3:-2] 对比图
plt.figure()

plt.plot(test_y[:, -3:-2][6:12].flatten(), marker='o', label='y_Real')
plt.plot(ypred[:, -3:-2][6:12].flatten(), marker='^', label="y_Predict")

plt.title("归一化相速Vpc")
plt.legend()
plt.show()
# [-2:-1] 对比图
plt.figure()
plt.plot(test_y[:, -2:-1][6:12].flatten(), marker='o', label='y_Real')
plt.plot(ypred[:, -2:-1][6:12].flatten(), marker='^', label="y_Predict")
plt.title("耦合阻抗Kc")
plt.legend()
plt.show()
# [-1]  对比图
plt.figure()
plt.plot(test_y[:, -1][6:12].flatten(), marker='o', label='y_Real')
plt.plot(ypred[:, -1][6:12].flatten(), marker='^', label="y_Predict")
plt.title("衰减常数")
plt.legend()
plt.show()
"""   =======    [6:12] ======= """
#[6:12]
print('MSE 归一化相速[6:12] \n', mean_squared_error(test_y[:, -3:-2][6:12].flatten(), ypred[:, -3:-2][6:12].flatten()))
print('MSE 耦合阻抗[6:12] \n', mean_squared_error(test_y[:, -2:-1][6:12].flatten(), ypred[:, -2:-1][6:12].flatten()))
print('MSE 衰减常数[6:12] \n', mean_squared_error(test_y[:, -1][6:12].flatten(), ypred[:, -1][6:12].flatten()))

# 拼接数据
result = np.concatenate((test_y, ypred), axis=1)
print(result.shape)

df = pd.DataFrame(result)

pathexample_pandas = './data/example_pandas.xlsx'
df.to_excel(pathexample_pandas, index=False)

result = np.concatenate((xy, ypred), axis=1)
print(result.shape)

df = pd.DataFrame(result)
pathexample_pandas11 = './data/example_pandas11.xlsx'
df.to_excel(pathexample_pandas11, index=False)

