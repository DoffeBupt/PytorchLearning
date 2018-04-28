from torch.autograd import Variable
import torch
if __name__=="__main__":
    x = Variable(torch.ones(2, 2), requires_grad = True)
    y = x + 2


    # y 是作为一个操作的结果创建的因此y有一个creator
    z = y * y * 3
    w = z * z * 3

    out = z.mean()
    print(out)
    # 现在我们来使用反向传播
    out.backward()

    # out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
    # 在此处输出 d(out)/dx
    print(x.grad)