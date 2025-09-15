# import Krigingtest
from test_pyx import *
import os
import sys
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

if __name__ == "__main__":
    # datainputPath = sys.argv[5]
    # dataoutputPath = sys.argv[6]
    datainputPath = 'inputdata.txt'
    dataoutputPath = 'outputdata.txt'
    new_data_path = 'new_inputdata.txt'
    model_save_path = 'Kriging4Gun.pth'
    if os.path.exists(model_save_path):
        total_mse, mse_per_column, y_test_real, y_pred_real, evals_result, X, y = train_and_predict(datainputPath,
                                                                                                    dataoutputPath,
                                                                                                    model_save_path)
        print(f'Loaded and predicted with saved models.')
    else:
        total_mse, mse_per_column, y_test_real, y_pred_real, evals_result, X, y = train_and_predict(datainputPath,
                                                                                                    dataoutputPath,
                                                                                                    model_save_path)
        plot_loss_curve(evals_result, datainputPath, dataoutputPath)
        print(f'Trained and saved new models.')

    write_mse_to_file(total_mse, mse_per_column, dataoutputPath)
    plot_predictions(y_test_real, y_pred_real, mse_per_column, dataoutputPath, start_idx=0, end_idx=100)

    # 对新数据进行预测
    if os.path.exists(model_save_path):
        predictions = predict_new_data(new_data_path, model_save_path)
        formatted_predictions = format_result(predictions)
        save_result_to_file(formatted_predictions, 'predictions.txt')
        print('Predictions for new data saved to new_predictions.txt')
