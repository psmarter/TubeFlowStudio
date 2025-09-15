import Krigingtest
# from test_pyx import *
import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
        # labels, output_data = Krigingtest.read_txt_file(output_data_path)
        # labels, output_data = read_txt_file(output_data_path)
        # for label, mse_col in zip(labels, mse_columns):
        #     f.write(f"{label} MSE: {mse_col}\n")
        labels, output_data = Krigingtest.read_txt_file(output_data_path)
        if len(labels) == 1:
            f.write(f"{labels[0]} MSE: {mse_columns[0]}\n")
        else:
            for label, mse_col in zip(labels, mse_columns):
                f.write(f"{label} MSE: {mse_col}\n")

def plot_loss_curve(evals_result, input_data_path, output_data_path, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure()
    plt.plot(evals_result, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.title('总训练损失曲线')
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
    # datainputPath = 'inputdata.txt'
    # dataoutputPath = 'outputdata.txt'
    #new_data_path = 'new_inputdata.txt'
    # model_save_path = 'NewModel.pth'
    # if os.path.exists(model_save_path):
    #     total_mse, mse_per_column, y_test_real, y_pred_real, evals_result, X, y = Krigingtest.train_and_predict(datainputPath,
    #                                                                                                 dataoutputPath,
    #                                                                                                 model_save_path)
    #     print(f'Loaded and predicted with saved models.')
    # else:
    total_mse, mse_per_column, y_test_real, y_pred_real, evals_result, X, y = Krigingtest.train_and_predict(datainputPath,
                                                                                                dataoutputPath,
                                                                                                model_save_temp_path)
    plot_loss_curve(evals_result, datainputPath, dataoutputPath, jpg_save_path)
    #print(f'Trained and saved new models.')

    write_mse_to_file(total_mse, mse_per_column, dataoutputPath, mseoutput_path)

    # 将模型文件移动到包含中文字符的路径
    shutil.move(model_save_temp_path, model_save_path)
    #Krigingtest.plot_predictions(y_test_real, y_pred_real, mse_per_column, dataoutputPath, start_idx=0, end_idx=100)

    # # 对新数据进行预测
    # if os.path.exists(model_save_path):
    #     predictions = Krigingtest.predict_new_data(new_data_path, model_save_path)
    #     formatted_predictions = format_result(predictions)
    #     save_result_to_file(formatted_predictions, 'predictions.txt')
    #     print('Predictions for new data saved to new_predictions.txt')
