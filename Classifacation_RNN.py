# RNN 后者的循环会结果取决于前边的东西
# LSTM 长短期记忆，
# RNN 缺陷:梯度消失，梯度爆炸，后果:回忆不起来长久的记忆
# 分线内容如果很重要，会扔到主线，否则会丢掉，受控于输入记忆控制
# RNN分类
# 把图片认为是从上向下进行看，一行一维度，强行加入时间维度
#
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64         # 一批训练的数量
TIME_STEP = 28          # 一共28行，所以共走28步(height)
INPUT_SIZE = 28         # 一步吃进去28个东西
LR = 0.01               # learning rate
DOWNLOAD_MNIST = False   # set to True if haven't download the data


# Mnist digital dataset
train_data = dsets.MNIST(
    root='./mnist/',                    # 下载的位置
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # 把图片转换为tensor的形式
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)

# plot one example
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

# Data Loader for easy mini-batch return in training
# 批训练数据读取
train_loader = \
    torch.utils.data.DataLoader(
        dataset=train_data,     # 数据是train_data
        batch_size=BATCH_SIZE,  # 批大小
        shuffle=True)           # 打乱

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape (2000, 28, 28) value in range(0,1)
test_y = test_data.test_labels.numpy().squeeze()[:2000]    # covert to numpy array


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__() # 继承父类的构造函数

        # RNN一般不太好收敛,用LSTM会比较好
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,  # 一步吃进去多少个像素点
            hidden_size=64,         # 神经元!hidden层的不同权重的单元个数rnn hidden unit
            num_layers=1,           # number of rnn layer,一个layer里还有28个64单元的东西
            # 看数据的三个维度如何排列的,一般情况下要是Batch在第一个维度,那么就是True
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # 全连接层
        # 输入64个,输出10个结果
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 下边那个会返回两个东西
        # 一个是r_out,另一个是hidden state,表示当前层我的理解
        # h_n,h_c 对应 分线程,主线程的state
        # none那个下节会讲
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        # 选择最后一个时刻的output
        # [batch, time step, input] 第二个参数设置为-1,说明取最后一个
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
# 一维的标签,不再是(0,0,0,1,0,0)哪种的了
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    # 强行多传一个序号参数
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                             # batch y

        output = rnn(b_x)                               # rnn output
        loss = loss_func(output, b_y)                   # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
