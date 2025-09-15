# import Autoencodertest
from test_pyx import *
import sys
import os

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
        labels, output_data = read_txt_file(output_data_path)
        if len(labels) == 1:
            f.write(f"{labels[0]} MSE: {mse_columns[0]}\n")
        else:
            for label, mse_col in zip(labels, mse_columns):
                f.write(f"{label} MSE: {mse_col}\n")


def predict_new_data(model, new_data_path, scaler_input, scaler_output):
    _, new_data = read_txt_file(new_data_path)
    new_data_scaled = scaler_input.transform(new_data)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(new_data_tensor)
        predictions = torch.tensor(scaler_output.inverse_transform(predictions.numpy()), dtype=torch.float32)

    return predictions.numpy()

if __name__ == "__main__":
    datainputPath = 'inputdata.txt'
    dataoutputPath = 'outputdata.txt'
    new_data_path = 'new_inputdata.txt'  # 新的数据文件路径
    model_save_path = 'Autoencoder4HighFreMoredata.pth'
    # 检查模型文件是否存在
    if os.path.exists(model_save_path):
        input_header, input_data = read_txt_file(datainputPath)
        output_header, output_data = read_txt_file(dataoutputPath)
        input_size = input_data.shape[1]
        output_size = output_data.shape[1]
        model = Autoencoder(input_size, output_size)
        model.load_state_dict(torch.load(model_save_path))
        print(f'Model loaded from {model_save_path}')

        # 加载scaler以便进行数据预处理
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        scaler_input.fit(input_data)
        scaler_output.fit(output_data)

        # 对新数据进行预测
        predictions = predict_new_data(model, new_data_path, scaler_input, scaler_output)

        # 保存预测结果
        formatted_predictions = format_result(predictions)
        save_result_to_file(formatted_predictions, 'predictions.txt')
        print('Predictions saved to predictions.txt')
    else:
        # 如果不存在，则训练模型并保存
        X, y, mse, mse_columns, losses, y_test_real, y_pred_real = train_and_predict(datainputPath, dataoutputPath,
                                                                                     model_save_path)
        write_mse_to_file(mse, mse_columns, dataoutputPath)
        plot_loss_curve(losses)
        plot_predictions(y_test_real, y_pred_real, mse_columns, dataoutputPath, start_idx=0, end_idx=100)
    # X, y, mse, mse_columns, losses, y_test_real, y_pred_real = train_and_predict(datainputPath, dataoutputPath)
    # write_mse_to_file(mse, mse_columns, dataoutputPath)
    # plot_loss_curve(losses)
    # plot_predictions(y_test_real, y_pred_real, mse_columns, dataoutputPath, start_idx=0, end_idx=100)
