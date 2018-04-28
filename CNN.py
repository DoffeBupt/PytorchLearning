import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# .cuda()显卡加速
# Variable变量有cuda()属性
# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',                                # 保存路径
    train=True,                                     # Ture,训练集,False,测试集
    transform=torchvision.transforms.ToTensor(),    # 原始数据改变成Tensor格式,原始的是nparray,这里改为了Tensor以后并且数据均匀的归一化
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # 看超参数
)

# plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()                                          # 打印出来画面等等

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=2) # shufflee:乱序,numofworkers,线程?

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False) # 对比与上边的True,这里是False
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000].cuda()/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# print (torch.unsqueeze(test_data.test_data, dim=1))
# print (test_data.test_data)
# 升维并且转化为Variable，１表示在第二个维度加了一个维度
test_y = test_data.test_labels.cuda()[:2000]
# 取2000个


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层１
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            # 卷积层
            nn.Conv2d(
                in_channels=1,              # 每个点的维度
                out_channels=16,            # 16个不同权重的卷积核进行扫描，提取出了16个维度的特征丢到了下一层
                kernel_size=5,              # 5*5的像素的卷积核
                stride=1,                   # 跳跃的步长
                padding=2,                  # 周围围上一圈0,这样都好扫,大小为卷积核中心到边缘的长度padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            # 激活函数
            nn.ReLU(),                      # activation
            # 池化层
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            # (16,14,14)
            # 池化层，相当于把那个底不变高增加的柱子，每2*2个像素，取一个最大值，降低底面大小
            # 相对于MaxPool还有Average取中值
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
        # 全卷积层,输入为32*7*7,输出为10个结果
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                   # (batch,32,7,7)
        x = x.view(x.size(0), -1)           # 展平到(batch, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN().cuda()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x).cuda()   # batch x
        b_y = Variable(y).cuda()   # batch y

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # 每50步预测一次结果
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
            # cuda仿佛不支持可视化,先注释掉啦~
            # if HAS_SK:
            #     # Visualization of trained flatten layer (T-SNE)
            #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            #     plot_only = 500
            #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            #     labels = test_y.numpy()[:plot_only]
            #     plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.cuda().cpu().numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].cuda().cpu().numpy(), 'real number')