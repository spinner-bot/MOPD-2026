#数字图像识别 开发者：浪兮（hhu2524030232张锐寒）

"""
【开发者注】
    本工具旨在通过深度学习方法实现以下功能：
   1.自动读取、识别以csv格式输入的28*28图像像素信息
   2.构建训练集(train-data set)
   3.通过预先设计好的“单输出多类别”神经网络处理图像
   4.。。。（先干活，一会再写）
        ————浪兮（hhu2524030232张锐寒）
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# 服务器无图形界面，matplotlib/seaborn 如需绘图请加 matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def reset():
    data = pd.read_csv("手写数字识别.csv")
    print(f"文件读取完成，数据形状: {data.shape}")
    x,y = (data.iloc[1:,1:]),data.iloc[1:,0]

    print(f"读取到{data.shape[0]-1}组图像像素数据")
    size=float(input("请输入训练集(train-data set)占比(default:0.8)："))
    if size > 1 or size < 0:
        size = 0.8
        print("输入不正确，已采用default值:0.8")

    x = x.astype(np.float32).values
    y = y.astype(np.int64).values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1-size,random_state=42)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))

class NeuralNetwork(nn.Module):
    def __init__(self,i_dim,o_dim):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(i_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, o_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        output = self.linear3(x)
        return output

    #备用
    '''def predict_prob(self, x):
        output = self.forward(x)
        prob = torch.softmax(output, dim=1)
        return prob
'''

def train_model(train_dataset,i_dim,o_dim,epochs,a,b):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(i_dim,o_dim).to(device)
    torch.manual_seed(1)
    dataloader = DataLoader(train_dataset, batch_size=a, shuffle=True)
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=b)
    model.train()
    print(f"开始训练模型... batch:{a},learning rate:{b:.2f} device：{device}")
    for epoch in range(epochs): #注意epoch。epochs。epoch_setting是不同的！！！！！
        total_loss = 0.0
        start = time.time()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        end = time.time()
        time_elapsed = end - start
        print(f"Epoch[{epoch + 1}/{epochs}]已完成, 损失平均数：{avg_loss:.4f},用时：{time_elapsed:.2f}s")
    return model


def test(model, test_dataset,a):
    #和训练一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始测试模型... batch:{a} device：{device}")
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=a, shuffle=False) #打乱没有必要
    correct = 0
    total = 0
    #在“禁用梯度下降"的情形下：
    with torch.no_grad():
        for data, target in test_loader:
            #和之前一样
            data, target = data.to(device), target.to(device)
            logits = model(data)
            _, predicted = torch.max(logits, dim=1)
                #语法知识点：占位符skip不被使用到的内容。我的总结是：右侧单数据囊括左侧多数据时，可能用到这一技巧
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy100 = 100 * correct / total
    print("\n【测试结果】")
    print(f"测试集样本总数：{total}")
    print(f"正确预测数：{correct}")
    print(f"测试集准确率：{accuracy100:.2f}%")
    return accuracy100



train_dataset, test_dataset, i_dim, class_n = reset()
epoch_setting=int(input("请设定深度学习的轮数:"))
bs=int(input("请设定批处理大小："))
a_set=float(input("请设定学习率："))
trained_model = train_model(train_dataset, i_dim, class_n, epoch_setting,bs,a_set)
test(trained_model, test_dataset,bs)

'''
while(True):
    print("请选择一个index，并进入绘图：")
'''