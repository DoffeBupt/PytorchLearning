# 每一个时间点进行输出

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate


## 单纯的画图代码,画理想效果的
# # show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)    # float32 for converting torch FloatTensor
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE, # 一次丢一个数据(y轴数据)
            hidden_size=32,     # 一个hidden state 有多少个神经元
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1) # 32个神经元的输入,一个输出

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        # 我用我当前的记忆,以及当前的值,预测出下一个应该出的值,以及创造出新的记忆
        # h_state只有最新的记忆状态,而x,rout实际上是多维的数据,是一个拥有多个时间点的数组
        r_out, h_state = self.rnn(x, h_state)
        # ***上边这句话在更新当前的记忆
        # rout是一个每个元素拥有hiddensize个个数的数组
        # ***下边在更新outs,即每次预测的值
        outs = []    # save all predictions
        # r_out.size(1)即时间轴的大小
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            # 把自己的当前的时间的out压进去
            outs.append(self.out(r_out[:, time_step, :]))
        # torch.stack 相当于把他压成一个tensor
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

# 将一个空值赋值给第一个h_state用来迭代
h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(600):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)
    # 打包
    # x输入,y输出
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    # 再用的时候记得用这个重新包装下
    # 否则会出BUG
    h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.02)

plt.ioff()
plt.show()
