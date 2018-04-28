import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math

n_data = torch.ones(1000, 2) #100列2行的1

x0 = torch.normal(2*n_data, 2)      # class0 x data (tensor), shape=(100, 2) 数据，前方为均值，后方为标准差
y0 = torch.zeros(1000)               # class0 y data (tensor), shape=(100, 1) 标签
x1 = torch.normal(-2*n_data, 2)     # class1 x data (tensor), shape=(100, 2) 数据
y1 = torch.ones(1000)                # class1 y data (tensor), shape=(100, 1) 标签
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer


x,y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy(), y.data.numpy()) #plot 连续 scatter 离散
#plt.show()

#搭建方法1
class Net(torch.nn.Module):
    def __init__(self, n_feature,n_hidden, n_output):
        super(Net, self).__init__()#调用父类的构造函数
        self.hidden0 = torch.nn.Linear(n_feature, n_hidden) #隐藏层
        self.hidden1 = torch.nn.Linear(n_hidden, n_hidden)  # 隐藏层
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output) #预测层
        pass
    #层的信息

    def forward(self, x): #x:输入信息
        x = F.relu(self.hidden0(x))#通过hidden层进行加工，再用激励函数进行激活
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
        x = self.predict(x)
        return x #部分需要激励函数截断
    #前向传递的过程

#搭建方法2：快速搭建
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2),
    #relu了第一层。。。
)

net = Net(2,50,2)#输入2->50->2
print(net)
#print(net)

plt.ion() #实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(),lr=0.01)#SGD，一个常用的优化器,lr<1学习效率，越高越快越粗糙
loss_func = torch.nn.CrossEntropyLoss()#计算的是多分类的概率，每个类型和标签的概率，然后算这种形式的误差 [0,0,1][0.1,0.1,0.8]

for t in range(200):
    out = F.softmax(net(x)) #输出的是概率，通过softmax(out)转化为概率
    print(out)#softmax,归一化到[0,1]
    loss = loss_func(out,y)#约定前方是预测值
    #print(loss)
    optimizer.zero_grad()#先将每次的梯度置零
    loss.backward()#再开始每次的反向传递，计算梯度，反传播到Variable
    optimizer.step()#以学习效率0.5优化
    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/2000.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff
plt.show()
