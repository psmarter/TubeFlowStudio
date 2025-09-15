import GPRtest
# from test_pyx import *
import sys
import matplotlib.pyplot as plt

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
        labels, output_data = GPRtest.read_txt_file(output_data_path)
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
    X, y, total_mse, mse_columns, evals_result, y_test_real, y_pred_real = GPRtest.train_and_predict(datainputPath,
                                                                                                     dataoutputPath)
    write_mse_to_file(total_mse, mse_columns, dataoutputPath)
    GPRtest.plot_loss_curve(evals_result, datainputPath, dataoutputPath)
    GPRtest.plot_predictions(y_test_real, y_pred_real, mse_columns, dataoutputPath)
