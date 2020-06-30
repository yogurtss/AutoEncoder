import numpy as np
import torch
import os
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split


def precess_data(file_path):
    file_path = file_path
    if not os.path.exists(file_path):
        raise FileNotFoundError()
    df = pd.read_csv(file_path)
    normal_events = df[df['Class'] == 0]
    anomaly_events = df[df['Class'] == 1]
    normal_data = normal_events[normal_events.columns[1:29]].to_numpy(dtype=np.float32)
    normal_label = normal_events[normal_events.columns[30]].to_numpy(dtype=np.float32)
    mean = np.mean(normal_data, 0)
    std = np.std(normal_data, 0)
    normal_data = (normal_data - mean) / std
    x_train, x_test, _, _ = train_test_split(normal_data, normal_label, train_size=0.7, test_size=0.3,
                                             random_state=99)
    x_anomaly = anomaly_events[anomaly_events.columns[1:29]].to_numpy(dtype=np.float32)
    return x_train, x_test, x_anomaly


def validate(model, file_path):
    _, x_test, x_anomaly = precess_data(file_path)
    mse = np.zeros(x_test.shape[0], dtype=np.float32)
    criterion = nn.MSELoss()
    model.eval()
    for i in tqdm(range(x_test.shape[0])):
        x = torch.tensor(x_test[i])
        x = x.unsqueeze(0)
        predict = model(x)
        loss = criterion(predict, x)
        mse[i] = loss.item()
    thres = np.percentile(mse, 95)
    print('The threshold  mse for anomaly events is {}'.format(thres))

    correct_num = 0
    total_num = x_anomaly.shape[0]

    # ano_precit = np.zeros(x_anomaly.shape, dtype=np.float32)

    for i in range(total_num):
        x_ano = torch.tensor(x_anomaly[i]).unsqueeze(0)
        predict = model(x_ano)
        loss = criterion(predict, x).item()
        if loss > thres:
            correct_num += 1

    print(correct_num / total_num)


if __name__ == '__main__':
    model_path = './result/2020-06-26-16/model_best.pth'
    state_dict = torch.load(model_path, map_location='cpu')
    state_pa = state_dict['model_state_dict']
    model = AutoEncoder(28)
    model.load_state_dict(state_pa)

    file_path = './data/creditcard.csv'

    validate(model, file_path)

