# DQN Q_Learning的神经网络版本

# 强化学习 分数导向
# Model Free: 不理解环境,直接从现实中获取
    # 所以需要等待现实世界的反馈
# Model_Based RL: 创建一个虚拟环境,对现实世界建模,从虚拟中想象出数据,获取数据

# 基于概率:不同概率的选择
# 基于价值:一定会选择最高价值的

# 对于连续动作,基于概率才可以使用

# 回合更新: 一局更新一次
# 单步更新: 一步更新一次,边玩边学

# 在线学习: 必须实时学习
# 离线学习: 可以看别人玩学习

# Q Lreaning
    # 根据一个Q表，state与决策共同决定Q值,取较大的值选择
    # Q现实=现有奖励+下一步预测的奖励*衰减系数
    # 不停地迭代 就是 现有奖励+下一步*衰减+下下步*衰减平方.......
    # 衰减系数取(0~1)
# 然而QLraening表格太大了
# 画个神经网络,用神经网络来算Q值好了
# 新NN = 老NN + 阿尔法(现实-估计)

# Experience Replay
    # 随机学习别人的经历,打乱相关性
# Fixed Q-targets
    # 打乱相关性 Q估计用最新的,Q现实用好久以前的

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount # 那个消减系数
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
# 导入gym的模拟场所
env = gym.make('CartPole-v0')
env = env.unwrapped
# 小车的动作
# 小车的观测
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# 神经网络的继承设置
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50) # 输入我现在的观测值
        self.fc1.weight.data.normal_(0, 0.1)   # 用正态分布随机生成初始的观测值
        self.out = nn.Linear(50, N_ACTIONS) # 输出我现在的观测值
        self.out.weight.data.normal_(0, 0.1)   # 同initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # 激流函数
        actions_value = self.out(x) # 计算每一个动作的价值
        return actions_value

# DQN的
class DQN(object):
    def __init__(self):
        # 两个基本一样的神经网络,二者有一个延迟的效果
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0 # 学习步数计数器                                # for target updating
        self.memory_counter = 0     # 学习记忆计数器                                         # for storing memory
        # 用全0初始化记忆库
        # 两倍的State数值以及a,r
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # 采取动作的
    def choose_action(self, x):
        # 输入观测值
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        # 随机选取的概率
        # 一定概率选取最优解
        if np.random.uniform() < EPSILON:   # 贪婪,选取高价值动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            # 随机选取了一个动作
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # 记忆库,存储学习的过程
    # 存储状态,动作,奖励,预测的下一个状态
    def store_transition(self, s, a, r, s_):
        # 捆在一起
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        # 索引值在超过记忆最大值以后会覆盖掉最老的记忆
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        # 检测是否要更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 学了TRI步(超参数定义为100),就更新TargetNet
            # 将evalnet赋值到targetnet
            # evalnet每一步都在更新,但是targetnet时不时在更新
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        # 从记忆库(32个)中随机抽取记忆
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        # 几个属性分开存储
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        # q_eval w.r.t the action in experience
        # eval_net(b_s)会给出当前动作下所有动作的价值,然后在之中选取当前选择了的动作的价值
        # q_eval即这一次我动作的价值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        # 下一个状态的状态的价值,且不进行反向传播
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        # q_target未来的价值中最大的那个
        # 最大值是第一个数,索引是第二个数
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 反向传播blabla
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')

# 强化学习的过程
for i_episode in range(400):
    # 所处的状态
    s = env.reset() # env = 小车模型
    ep_r = 0
    while True:
        # 环境渲染下,确认现在所处的环境
        env.render()
        # 根据现在所处的状态,采取一个行为
        a = dqn.choose_action(s)

        # take action
        # 环境根据我的行为,给我现在的反馈
        s_, r, done, info = env.step(a)

        # modify the reward
        # reward 进行了修改
        # 车往中间reward大,立住杆子reward大
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        # dqn存储我现在的反馈:[当前]之前状态,施加动作,环境给我的奖励,环境导引我的下一个状态
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            # 学习
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        # 回合结束,我就去下一个回合
        if done:
            break

        # 现在的状态被更新为下一个引导的状态
        s = s_
