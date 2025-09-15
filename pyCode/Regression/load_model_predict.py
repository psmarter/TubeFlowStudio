import XGBoosttest
# import CNNtest
# import GPRtest
# import Krigingtest
# import CNN_xgboosttest
# import Autoencodertest
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

if __name__ == "__main__":
    PredictDatapath = 'PredictData.txt'
    modelpath = 'XGBoost4Collector.pth'
    modeltype = 'XGBoost'
    if modeltype == 'XGBoost':
        if os.path.exists(modelpath):
            predictions = XGBoosttest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')
    elif modeltype == 'CNN':
        if os.path.exists(modelpath):
            predictions = CNNtest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')
    elif modeltype == 'GPR':
        if os.path.exists(modelpath):
            predictions = GPRtest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')
    elif modeltype == 'Kriging':
        if os.path.exists(modelpath):
            predictions = Krigingtest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')
    elif modeltype == 'CNN_xgboost':
        if os.path.exists(modelpath):
            predictions = CNN_xgboosttest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')
    elif modeltype == 'Autoencoder':
        if os.path.exists(modelpath):
            predictions = Autoencodertest.predict_new_data(PredictDatapath, modelpath)
            formatted_predictions = format_result(predictions)
            save_result_to_file(formatted_predictions, 'PredictResult.txt')
            print('Predictions for new data saved to new_predictions.txt')