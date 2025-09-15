import Autoencodertest
# from test_pyx import *
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import StandardScaler
import shutil


def format_result(data):
    formatted_data = []
    for sample in data:
        formatted_sample = ' '.join(map(str, sample))
        formatted_data.append(formatted_sample)
    return '\n'.join(formatted_data)

def save_result_to_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(data)
    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def write_mse_to_file(mse, mse_columns, output_data_path, file_name='mseoutput.txt'):
    with open(file_name, 'w') as f:
        f.write(f"Total MSE: {mse}\n")
        labels, output_data = Autoencodertest.read_txt_file(output_data_path)
        if len(labels) == 1:
            f.write(f"{labels[0]} MSE: {mse_columns[0]}\n")
        else:
            for label, mse_col in zip(labels, mse_columns):
                f.write(f"{label} MSE: {mse_col}\n")

def plot_loss_curve(evals_result, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    plt.plot(evals_result, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    datainputPath = sys.argv[1]
    dataoutputPath = sys.argv[2]
    model_save_path = sys.argv[3]
    jpg_save_path = sys.argv[4]
    mseoutput_path= sys.argv[5]

    # 获取当前 Python 文件的所在目录
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # 设置模型保存的临时路径
    model_save_temp_path = os.path.join(current_directory, 'temp_model.pth')

    # datainputPath = 'C:\\Users\\s2\\Desktop\\DMDev\\bin\\NT_VC16_64_DLLD\\scripts\\inputdata.txt'
    # dataoutputPath = 'C:\\Users\\s2\\Desktop\\DMDev\\bin\\NT_VC16_64_DLLD\\scripts\\outputdata.txt'
    # model_save_path = 'C:\\Users\\s2\\Desktop\\MTSS_test\\1017x1\\电子枪1\\model.gunpth'
    # jpg_save_path = 'C:\\Users\\s2\\Desktop\\DMDev\\bin\\NT_VC16_64_DLLD\\runtime\LossCurve.jpg'
    # mseoutput_path = 'C:\\Users\\s2\\Desktop\\DMDev\\bin\\NT_VC16_64_DLLD\\runtime\mseoutput'
    # datainputPath = 'inputdata.txt'
    # dataoutputPath = 'outputdata.txt'
    # new_data_path = 'new_inputdata.txt'  # 新的数据文件路径
    # model_save_path = 'NewModel.pth'
    # 检查模型文件是否存在
    # if os.path.exists(model_save_path):
    #     input_header, input_data = Autoencodertest.read_txt_file(datainputPath)
    #     output_header, output_data = Autoencodertest.read_txt_file(dataoutputPath)
    #     input_size = input_data.shape[1]
    #     output_size = output_data.shape[1]
    #     model = Autoencodertest.Autoencoder(input_size, output_size)
    #     model.load_state_dict(torch.load(model_save_path))
    #     print(f'Model loaded from {model_save_path}')
    #
    #     # 加载scaler以便进行数据预处理
    #     scaler_input = StandardScaler()
    #     scaler_output = StandardScaler()
    #     scaler_input.fit(input_data)
    #     scaler_output.fit(output_data)
    #
    #     # 对新数据进行预测
    #     predictions = Autoencodertest.predict_new_data(model, new_data_path, scaler_input, scaler_output)
    #
    #     # 保存预测结果
    #     formatted_predictions = format_result(predictions)
    #     save_result_to_file(formatted_predictions, 'predictions.txt')
    #     print('Predictions saved to predictions.txt')
    # else:
        # 如果不存在，则训练模型并保存
    X, y, mse, mse_columns, losses, y_test_real, y_pred_real = Autoencodertest.train_and_predict(datainputPath, dataoutputPath,
                                                                                 model_save_temp_path)
    write_mse_to_file(mse, mse_columns, dataoutputPath, mseoutput_path)
    plot_loss_curve(losses, jpg_save_path)
    Autoencodertest.plot_predictions(y_test_real, y_pred_real, mse_columns, dataoutputPath, start_idx=0, end_idx=100)

    # 将模型文件移动到包含中文字符的路径
    shutil.move(model_save_temp_path, model_save_path)

    # X, y, mse, mse_columns, losses, y_test_real, y_pred_real = train_and_predict(datainputPath, dataoutputPath)
    # write_mse_to_file(mse, mse_columns, dataoutputPath)
    # plot_loss_curve(losses)
    # plot_predictions(y_test_real, y_pred_real, mse_columns, dataoutputPath, start_idx=0, end_idx=100)
