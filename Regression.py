import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math

x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)#un 二维化,把一个长度为100的改为面积为1*100
print(x.shape)

y = x.pow(2) #+ 0.2*torch.rand(x.size())#平方，噪点

x,y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy(), y.data.numpy()) #plot 连续 scatter 离散
#plt.show()

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

net = Net(1,50,1)#输入1->10->1
print(net)
#print(net)

plt.ion() #实时打印
plt.show()

optimizer = torch.optim.Adam(net.parameters(),lr=0.01)#SGD，一个常用的优化器,lr<1学习效率，越高越快越粗糙
loss_func = torch.nn.MSELoss()#误差，均方差MeanSquareError，分类误差用另一个算

for t in range(2000):
    prediction = net(x)

    loss = loss_func(prediction,y)#约定前方是预测值

    optimizer.zero_grad()#先将每次的梯度置零
    loss.backward()#再开始每次的反向传递，计算梯度
    optimizer.step()#以学习效率0.5优化
    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)
        plt.text(0.5, 0, 'Loss=%4f' % loss.data[0], fontdict={'size':20, 'color': 'red'})
        plt.pause(0.05)

plt.ioff
plt.show()
