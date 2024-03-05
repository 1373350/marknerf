from coder import model
from torch import nn
import torch
from train import flat_source_size,flat_payload_size
#后面开始对训练过程开始优化
model.cuda()

criterion = nn.MSELoss() #均方误差，是预测值与真实值之差的平方和的平均值
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #优化
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)#第一个参数就是所使用的优化器对象
#第二个参数就是每多少轮循环后更新一次学习率(lr)
#第三个参数就是每次更新lr的gamma倍

metric = nn.L1Loss() #损失函数nn.L1Loss()   作用其实就是计算网络输出与标签之差的绝对值，返回的数据类型可以是张量，也可以是标量。


# 定义一个函数来获取损失值
def get_loss(model, s, p):
    e_out, dp_out = model.forward((s, p))

    e_loss = criterion(e_out.contiguous().view((-1, flat_source_size)), s.contiguous().view((-1, flat_source_size)))
    # 计算编码器输出图像和源图像的均方误差
    #     e_sum_loss = criterion(torch.div(torch.sum(e_out, dim=1).view((-1, int(flat_image_size / 3.))), 765) , torch.div(torch.sum(s, dim=-1).view((-1, int(flat_image_size / 3.))), 765))
    dp_loss = criterion(dp_out.contiguous().view((-1, flat_payload_size)), p.contiguous().view((-1, flat_payload_size)))
    # 计算解码器输出秘密图像和源秘密图像的均方误差
    ds_loss = criterion(ds_out.contiguous().view((-1, flat_source_size)), s.contiguous().view((-1, flat_source_size)))
    # 计算解码器输出源图像与源图像之间的均方误差
    loss = e_loss + dp_loss + ds_loss  # + e_sum_loss
    # 计算均方误差之和
    return loss
def train_step(model, optimizer, s, p):
    temp_loss = 0.#临时损失初始值为零
    optimizer.zero_grad()#梯度清零

    loss = get_loss(model, s, p)#获取损失和

    loss.backward()#方向传播求梯度
    optimizer.step()#    进行单次优化，所有的optimizer都实现了step()方法，这个方法会更新所有的参数。它能按两种方式来使用：
#1.optimizer.step(),这是大多数optimizer所支持的简化版本。一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。
#简化版深度学习流程for input, target in dataset：从数据集导入输入和目标
#                optimizer.zero_grad()  梯度清零
#                output = model(input)  通过定义好的模型求得输出
#                loss = loss_fn(output, target)  计算输出和目标的损失
#                loss.backward()  方向传播求梯度
#                optimizer.step()  进行单次优化参数的值
#                optimizer.step(closure) 关闭函数
#一些优化算法例如Conjugate Gradient和LBFGS需要重复多次计算函数，因此你需要传入一个闭包去允许它们重新计算你的模型。这个闭包应当清空梯度，计算损失，然后返回。
#例子：for input, target in dataset:
#    def closure():
#        optimizer.zero_grad()
#        output = model(input)
#        loss = loss_fn(output, target)
#        loss.backward()
#        return loss
#    optimizer.step(closure)
    temp_loss += loss.item() #item()将一个零维张量转换成浮点数 ，因为一个epochs里也是按照很多个batchs进行训练。
   # 所以需要把一个epochs里的每次的batchs的loss加起来，等这一个epochs训练完后，会把累加的loss除以batchs的数量，得到这个epochs的损失。
    return temp_loss#临时损失
