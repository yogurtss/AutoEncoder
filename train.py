import os
import pandas as pd
import torch
import numpy as np
from math import floor
from tqdm import tqdm
import torch.nn as nn
import datetime
import argparse
from dataset.credit_dataset import CreditDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from models.auto_encoder import AutoEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = './result/{}'.format(nowTime)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def valiadate(model, criterion, val_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for i, (x) in enumerate(val_loader):
            output = model(x)
            loss = criterion(output, x)
            total_loss += loss.item()
    return total_loss


def save_model(model, optimizer, step, path):
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, path)


def cut_data(file_path, ratio=[0.7, 0.1, 0.2]):
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
    x_train, x_test, _, _ = train_test_split(normal_data, normal_label, train_size=0.7, test_size=0.3, random_state=99)
    return x_train, x_test


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=False,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--loss', type=str, default="L2",
                        help="L1 or L2")
    parser.add_argument('--data_dir', type=str, default='./data/creditcard.csv',
                        help="Path of data file")
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    parser.add_argument("--load_model", type=str, default='result/2020-03-08-00/model_best.pth')
    return parser.parse_args()


def main():
    args = cfg()
    writer = SummaryWriter(result_dir)
    args.load = False

    x_train, x_test = cut_data(args.data_dir)
    train_dataset = CreditDataset(x_train)
    val_dataset = CreditDataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoEncoder(28)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    cy_len = floor(len(train_dataset) / args.batch_size // 2)
    clr = lr_scheduler.CyclicLR(optimizer, args.lr, args.max_lr, cy_len, cycle_momentum=False)

    criterion = nn.MSELoss()
    state = {"step": 0,
             "worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf}

    while state["worse_epochs"] < args.hold_step:
        print("Training one epoch from iteration " + str(state["step"]))
        model.train()
        for i, (x) in enumerate(train_loader):
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar("learning_rate", cur_lr, state['step'])
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, x)
            loss.backward()
            writer.add_scalar("training_loss", loss, state['step'])
            optimizer.step()
            clr.step()
            # clr.step()
            state['step'] += 1

            if i % 50 == 0:
                print(
                    "{:4d}/{:4d} --- Loss: {:.6f}  with learnig rate {:.6f}".format(
                        i, len(train_dataset) // args.batch_size, loss, cur_lr))

        val_loss = valiadate(model, criterion, val_loader)
        # val_loss /= len(val_dataset)//args.batch_size
        print("Valiadation loss" + str(val_loss))
        writer.add_scalar("val_loss", val_loss, state['step'])

        writer.add_scalar("val_loss", val_loss, state["step"])

        # EARLY STOPPING CHECK
        checkpoint_path = args.model_path + str(state['step']) + '.pth'
        print("Saving model...")
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
            best_checkpoint_path = args.model_path + 'best.pth'
            save_model(model, optimizer, state, best_checkpoint_path)
        print(state)
        state["epochs"] += 1
        if state["epochs"] % 5 == 0:
            save_model(model, optimizer, state, checkpoint_path)
    last_model = args.model_path + 'last_model.pth'
    save_model(model, optimizer, state, last_model)
    print("Training finished")
    writer.close()


if __name__ == '__main__':
    main()
