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

if __name__ == "__main__":
    PredictDatapath = 'new_inputdata.txt'
    modelpath = 'CNN4HighFreMoredata.pth'
    datainputPath = 'CNNinputdata.txt'
    dataoutputPath = 'CNNoutputdata.txt'
    if os.path.exists(modelpath):
        input_header, input_data = read_txt_file(datainputPath)
        output_header, output_data = read_txt_file(dataoutputPath)
        input_size = input_data.shape[1]
        output_size = output_data.shape[1]

        model = CNN1D(input_size, output_size)
        model.load_state_dict(torch.load(modelpath))
        print(f'Model loaded from {modelpath}')

        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        scaler_input.fit(input_data)
        scaler_output.fit(output_data)

        predictions = predict_new_data(model, PredictDatapath, scaler_input, scaler_output)

        formatted_predictions = format_result(predictions)
        save_result_to_file(formatted_predictions, 'PredictResult.txt')
        print('Predictions saved to predictions.txt')