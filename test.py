import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

device = ['cuda:0' if torch.cuda.is_available() else 'cpu'][0]


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=11, out_features=1, dtype=float),
            # nn.Sigmoid(),
            # nn.Linear(in_features=32, out_features=1, dtype=float)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, fa, fb):
        one = torch.ones(fa.shape).to(device)
        return one - F.sigmoid(fb - fa)


class PairWiseDataset(Dataset):
    def __init__(self, df):
        # self.df = df.drop(labels=['seq'], axis=1)
        # self.df = torch.from_numpy(self.df.values)
        # self.index = []
        # n = self.df.shape[0]
        # print(self.df.shape)
        # print(n)
        # for i in range(n):
        #     for j in range(i+1, n):
        #         self.index.append([i, j])
        #
        # self.y = torch.from_numpy(df['seq'].values)
        self.data = torch.from_numpy(df.values)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # i = self.index[item]
        # x1 = self.df[i[0]]
        # x2 = self.df[i[1]]
        # y1 = self.y[i[0]]
        # y2 = self.y[i[1]]
        # return x1, x2, y1, y2
        x = self.data[item]
        x1 = x[:11]
        x2 = x[11:]
        return x1, x2

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        WEIGHT = os.path.join(self.save_path, 'best_weights.pth')
        MODEL = os.path.join(self.save_path, 'best_model.pth')
        torch.save(model.state_dict(), WEIGHT)  # 这里会存储迄今最优模型的参数
        torch.save(model, MODEL)
        self.val_loss_min = val_loss


class NDCGTool:
    def __init__(self, rel):
        self.rel = rel

    def DCG(self):
        res = 0.0
        for i, r in enumerate(self.rel):
            # i从0开始，DCG计算中的i从1开始，故+2
            lg2 = math.log2(i + 2)
            res += r / lg2

        return res

    def IDCG(self):
        res = 0.0
        sorted_rel = sorted(self.rel, reverse=True)
        for i, r in enumerate(sorted_rel):
            lg2 = math.log2(i + 2)
            res += r / lg2

        return res

    def NDCG(self):
        return self.DCG() / self.IDCG()


def evalution_fun(df_name, model, criterion, mode):
    if mode == 0:
        df = pd.read_csv(f'data/data_4_2/tweet_id_valid_normalize/{df_name}')
    elif mode == 1:
        df = pd.read_csv(f'data/data_4_2/tweet_id_test_normalize/{df_name}')
        # df = pd.read_csv(f'data/tweet_id_test/{df_name}')

    # df = pd.read_csv('data/tweet_id/1333846411985739776.csv')
    # std = df.std()
    # print(std)
    test_ds_x = df.drop(labels=['seq', 'o_tid'], axis=1)
    test_ds_x = torch.from_numpy(test_ds_x.values)
    test_ds_x = test_ds_x.to(device)
    tot = test_ds_x.shape[0]
    model.eval()
    score_list = []
    with torch.no_grad():
        for i in range(tot):
            score = model(test_ds_x[i])
            score_list.append([score.tolist()[0], i])

    valid_tot_loss = 0.0
    batch_num = 0
    for i in range(tot):
        for j in range(i + 1, tot):
            batch_num += 1
            fa = torch.tensor(score_list[i][0])
            fb = torch.tensor(score_list[j][0])
            loss = criterion(fa, fb)
            valid_tot_loss += loss.item()

    # print(score_list)
    # print("排序")
    score_list = sorted(score_list)
    # print(score_list)

    evalution = 0.0
    # NDCG的相关性分数
    rel = []
    for i, s in enumerate(score_list):
        evalution += (i - s[1]) ** 2
        # 以预测排名和实际排名的差值绝对值+1(避免0)的倒数作为相关性分数
        rel.append(1 / (math.fabs(i - s[1]) + 1))

    ndcg = NDCGTool(rel)
    NDCG_score = ndcg.NDCG()

    # print(evalution/tot)
    return evalution / tot, valid_tot_loss, batch_num, NDCG_score, score_list




if __name__ == '__main__':

    # os.chdir('评论排序研究')
    print(os.getcwd())
    # 训练集
    # train_file_names = os.listdir('./data/tweet_id_2')
    train_data_path = './data/data_4_2/tweet_id_train_normalize'
    train_file_names = os.listdir(train_data_path)
    # 测试集
    # test_file_names = os.listdir('./data/tweet_id_test')
    test_data_path = './data/data_4_2/tweet_id_test_normalize'
    test_file_names = os.listdir(test_data_path)
    # 验证集
    # valid_file_names = os.listdir('./data/tweet_id_valid')
    valid_data_path = './data/data_4_2/tweet_id_valid_normalize'
    valid_file_names = os.listdir(valid_data_path)

    # 超参数
    lr = 1e-5
    epochs = 30
    batch_size = 16
    train_rate = 0.1
    print("加载数据")
    df = pd.read_csv('data/data_4_2/all_train_data.csv', header=None)
    # train_len = int(train_rate * len(df))

    tot = len(df)
    print(tot)

    train_ds = PairWiseDataset(df)
    trainLoader = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=4)
    # test_ds = PairWiseDataset(df[train_len:])
    # testLoader = DataLoader(test_ds, shuffle=True, batch_size=1)

    print(device)

    model_path = './model_normalize_accelerate'
    # model = MyModel()
    # model = torch.load('./model/model_model.pth')
    # model = torch.load('./model_1/model_model.pth')
    # model = torch.load('./model_E5_score/model_model.pth')
    # model = torch.load('./model_E5_merge_only/model_model.pth')
    # model = torch.load(model_path+'/model_model.pth', map_location=device)
    model = torch.load(model_path+'/best_model.pth', map_location=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MarginRankingLoss()
    criterion = Myloss()
    # 早停工具
    earlyStopping = EarlyStopping(model_path)
    evalution_list = [[] for i in range(len(valid_file_names))]
    NDCG_list = [[] for i in range(len(valid_file_names))]
    for epoch in range(epochs):
        tot_loss = 0.0
        batch_num = 0
        file_count = 0

        count = 0
        print("epoch")
        for i, data in enumerate(trainLoader, 0):
            x1, x2 = data
            # print(x1)
            # print(x2)
            x1 = x1.to(device)
            x2 = x2.to(device)
            optimizer.zero_grad()
            pred_y_1 = model(x1)
            pred_y_2 = model(x2)
            pred_y_1 = pred_y_1.to(device)
            pred_y_2 = pred_y_2.to(device)
            loss = criterion(pred_y_1, pred_y_2)
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            tot_loss += loss.sum()
            count += 1
            batch_num += loss.shape[0]
            print(f'当前训练数据组数{batch_num}/{tot}')

        print(f'epoch = {epoch}  loss = {tot_loss / batch_num}')
        with open(model_path + '/loss.txt', 'a', encoding='utf-8') as l_f:
            l_f.write(f'{tot_loss / batch_num}\n')

        # 验证集总loss
        valid_tot_loss = 0.0
        # 验证集总数据条数（数据对）
        valid_tot_batch = 0

        for i, f_n in enumerate(valid_file_names):
            # 三个返回值分别为评估值（均方差），该帖子评论的总loss，数据对总数
            evo, valid_tot_loss_file, valid_batch_num, NDCG_score, scores_list = evalution_fun(f_n, model, criterion, 0)
            valid_tot_loss += valid_tot_loss_file
            valid_tot_batch += valid_batch_num
            evalution_list[i].append(evo)
            NDCG_list[i].append(NDCG_score)
            print(f'当前评估文件数: {i + 1}/{len(valid_file_names)}')

        with open(model_path + '/NDCG.txt', 'w', encoding='utf-8') as ndcg_f:
            for i, l in enumerate(NDCG_list):
                ndcg_f.write(f'{i},')
                for item in l:
                    ndcg_f.write(f'{item},')
                ndcg_f.write('\n')

        with open(model_path + '/valid_loss.txt', 'a', encoding='utf-8') as vl_f:
            vl_f.write(f'{valid_tot_loss / valid_tot_batch}\n')

        with open(model_path + '/evalution.txt', 'w', encoding='utf-8') as e_f:
            for i, l in enumerate(evalution_list):
                e_f.write(f'{i},')
                for item in l:
                    e_f.write(f'{item},')
                e_f.write('\n')

        WEIGHT = model_path + '/model_weights.pth'
        MODEL = model_path + '/model_model.pth'
        torch.save(model.state_dict(), WEIGHT)
        torch.save(model, MODEL)
        print('模型保存完成！')
        # 早停
        earlyStopping(valid_tot_loss / valid_tot_batch, model)
        if earlyStopping.early_stop:
            print("=============Early stopping============")
            break

        print(f'epoch = {epoch} 评估结束！')
