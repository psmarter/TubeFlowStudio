import XGBoosttest
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
        labels, output_data = XGBoosttest.read_txt_file(output_data_path)
        # for label, mse_col in zip(labels, mse_columns):
        #     f.write(f"{label} MSE: {mse_col}\n")
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
    # test_y, ypred, mse, mse_columns, models, evals_result, x_data, y_data = XGBoosttest.train_and_predict('input_data.txt', 'output_data.txt')
    test_y, ypred, mse, mse_columns, models, evals_result, x_data, y_data = XGBoosttest.train_and_predict(
        datainputPath, dataoutputPath)
    print(x_data.shape)
    print(y_data.shape)
    #print(evals_result)
    # XGBoosttest.plot_loss_curve(evals_result, datainputPath, dataoutputPath)
    XGBoosttest.plot_loss_curve(evals_result, datainputPath, dataoutputPath)
    #save_result_to_file(str(mse), 'mseoutput.txt')
    write_mse_to_file(mse, mse_columns, dataoutputPath)
    # XGBoosttest.plot_predictions(test_y, ypred, dataoutputPath)
    XGBoosttest.plot_predictions(test_y, ypred, mse_columns, dataoutputPath)
    # formatted_x_data = format_result(x_data)
    # save_result_to_file(str(formatted_x_data), 'inputdata.txt')
    # formatted_y_data = format_result(y_data)
    # save_result_to_file(str(formatted_y_data), 'outputdata.txt')
