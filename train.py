from coder import model
from attention import model1
from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from payloadimage import payload_train
from sourceimage import source_train
criterion = nn.MSELoss() #标准是损失函数均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#优化算法，学习率为0.001
model.to(device)
model1.to(device)
#model.parameters()保存的是Weights和Bais参数的值。
#class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
#    params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
#    lr (float, 可选) – 学习率（默认：1e-3）
#    betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
#    eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
#    weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
# 个人理解：
#lr：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值（如 0.3）在学习率更新前会有更快的初始学习，
#而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
#betas = （beta1，beta2）
#beta1：一阶矩估计的指数衰减率（如 0.9）。
#beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
#eps：epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。
metric = nn.L1Loss()#创建一个标准来测量输入x和目标y中每个元素之间的平均绝对误差（MAE）
# 训练
epochs = 50000# 迭代10000次
train_losses, val_losses = [], []
# 训练集损失，训练过程中的测试集损失

# batch_size：表示单次传递给程序用以训练的数据（样本）个数。比如我们的训练集有1000个数据。
# 这是如果我们设置batch_size=100，那么程序首先会用数据集中的前100个参数，即第1-100个数据来训练模型。
# 当训练完成后更新权重，再使用第101-200的个数据训练，直至第十次使用完训练集中的1000个数据后停止。
# 那么为什么要设置batch_size呢？
# 优势：
# 可以减少内存的使用，因为我们每次只取100个数据，因此训练时所使用的内存量会比较小。
# 这对于我们的电脑内存不能满足一次性训练所有数据时十分有效。可以理解为训练数据集的分块训练。
# 提高训练的速度，因为每次完成训练后我们都会更新我们的权重值使其更趋向于精确值。所以完成训练的速度较快。
# 劣势：
# 使用少量数据训练时可能因为数据量较少而造成训练中的梯度值较大的波动。


flat_source_size = 256 * 256 * 3
flat_payload_size =256 * 256 * 3
# valid_loss_min = +np.inf

for epoch in range(epochs):  # 一个循环
    model.train()  # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
    # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    train_loss = 0.
    s, p = source_train, payload_train
    # s为source_train中第0个元素到第64个元素的数据，p为 payload_train中第0个元素到第64个元素的数据
    s.to(device)
    p.to(device)
    # torch.from_numpy() 是将 numpy 转成 tensor ,如果是tensor转numpy，则直接 tensor_data.numpy()
    # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。调用model.cuda()，可以将模型加载到GPU上去.
    # 这种方法不被提倡，而建议使用model.to(device)的方式，这样可以显示指定需要使用的计算资源，特别是有多个GPU的情况下。
    optimizer.zero_grad()
    # 另外Pytorch 为什么每一轮batch需要设置optimizer.zero_grad：
    # 根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
    # 但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了。

    dp_out,ds_out= model.forward((s, p))  # 获取卷积层和relu层的结果encoder_output, decoded_payload, decoded_source

   # e_loss = criterion(e_out.contiguous().view((-1, flat_source_size)), s.contiguous().view((-1, flat_source_size)))
    dp_loss = criterion(dp_out.contiguous().view((-1, flat_payload_size)),p.contiguous().view((-1, flat_payload_size)))
    ds_loss = criterion(ds_out.contiguous().view((-1, flat_source_size)),s.contiguous().view((-1, flat_source_size)))
    loss = dp_loss + ds_loss

    loss.backward()  # 反向传播求梯度
    optimizer.step()
    # step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，
    # 所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
    # 注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。
    train_loss += loss.item()
    # item()用于在只包含一个元素的tensor中提取值，注意是只包含一个元素，否则的话使用.tolist()
    # 在训练时统计loss变化时，会用到loss.item()，能够防止tensor无限叠加导致的显存爆炸
    # train_loss 求一个batch.size的loss之和
    print("Train loss: ", train_loss )
path = "state_dict_model.pt"
torch.save(model.state_dict(), path)

