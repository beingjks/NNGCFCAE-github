import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import numpy as np
#import tensorflow as tf
#from tensorflow_probability import distributions



def train2(A, ll):
    batch_size = 128  # 批量大小
    # 500->50
    num_epochs =25# 训练轮数
    expect_tho = 0.05  # 期望的平均激活值，用于 KL 散度计算

    Arm = torch.FloatTensor(A)  # 将输入数据 A 转换为 PyTorch 张量，形状为 (39, 662)
    Arm = Arm.unsqueeze(0)  # 在第 0 维度上增加一个维度，形状变为 (1, 39, 662)
    Arm = Arm.unsqueeze(1)  # 在第 1 维度上增加一个维度，形状变为 (1, 1, 39, 662)

    emb = np.size(A, axis=1)  # 获取输入数据的特征维度，662
    m = np.size(A, axis=0)  # 获取输入数据的样本数量，39

    def KL_devergence(p, q):
        """
        计算两个分布 p 和 q 之间的 KL 散度
        :param p: 目标分布
        :param q: 模型输出的分布
        :return: KL 散度值
        """
        q = torch.nn.functional.softmax(q, dim=0)  # 对 q 进行 softmax 归一化
        q = torch.sum(q, dim=0) / batch_size  # 计算 q 的平均值
        s1 = torch.sum(p * torch.log(p / q))  # 计算 KL 散度的第一部分
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))  # 计算 KL 散度的第二部分
        return s1 + s2  # 返回 KL 散度值
   # model = ResidualSparseAutoEncoder(emb=emb)
    model = AutoEncoder(emb=emb)  # 初始化自编码器模型，输入维度为 emb=662
    #Optimizer = optim.Adam(model.parameters(), lr=0.1)  # 使用 Adam 优化器，学习率为 0.1
    Optimizer = optim.Adam(model.parameters(), lr=0.1)  # 使用 Adam 优化器，学习率为 0.1
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(32)])  # 创建一个长度为 32 的张量，每个元素都是 expect_tho=0.05
    _beta = 0.1  # KL 散度的权重

    for epoch in range(num_epochs):  # 训练循环，共 num_epochs 轮
        time_epoch_start = time.time()  # 记录当前 epoch 的开始时间
        l, decoder_out = model(Arm)  # 前向传播， 到达 forward函数 , 为啥是 4 维的 -> 假设输入数据的形状为 (batch_size, 1, height, width)。(数量，通道，高度，宽度)获取编码器的输出 l 和解码器的输出 decoder_out
        # 在这里转化即可  我这里需要的是二维，但是这个l是4 维
        # l 是（39，32）
        # decoder_out 是1，1，39，701 Arm 也是
        # attr_embedding (39,32)
        attr_embedding = l.detach().numpy()  # 将编码器的输出转换为 NumPy 数组，以便保存到文件


        # if ll == 1:  # 如果 ll 等于 1，保存嵌入结果到文件 "Data/attr embedding_m.txt"
        #     np.savetxt("Data/attr embedding_m.txt", attr_embedding)
        # elif ll == 0:  # 如果 ll 等于 0，保存嵌入结果到文件 "Data/attr embedding_d.txt"
        #     np.savetxt("Data/attr embedding_d.txt", attr_embedding)

        # if ll == 1:  # 如果 ll 等于 1，保存嵌入结果到文件 "Data/attr embedding_m.txt"
        #     np.savetxt("Data1/attr embedding_m.txt", attr_embedding)
        # elif ll == 0:  # 如果 ll 等于 0，保存嵌入结果到文件 "Data/attr embedding_d.txt"
        #     np.savetxt("Data1/attr embedding_d.txt", attr_embedding)

        if ll == 1:  # 如果 ll 等于 1，保存嵌入结果到文件 "Data/attr embedding_m.txt"
            np.savetxt("MDAD/attr embedding_m.txt", attr_embedding)
        elif ll == 0:  # 如果 ll 等于 0，保存嵌入结果到文件 "Data/attr embedding_d.txt"
            np.savetxt("MDAD/attr embedding_d.txt", attr_embedding)
        #
        # elif ll == 2:  # 如果 ll 等于 2，保存嵌入结果到文件 "Data/Sd_dis embedding.txt"
        #     np.savetxt("Data/Sd_dis embedding.txt", attr_embedding)
        # elif ll == 3:  # 如果 ll 等于 3，保存嵌入结果到文件 "Data/Sm_dis embedding.txt"
        #     np.savetxt("Data/Sm_dis embedding.txt", attr_embedding)

        loss = F.mse_loss(decoder_out, Arm)  # 计算重构误差（MSE Loss），即解码器的输出与输入之间的均方误差
        _kl = KL_devergence(tho_tensor, l)  # 计算 KL 散度，衡量编码器的输出与期望分布之间的差异
        loss = _beta * _kl + loss * (1 - _beta)  # 总损失 = KL 散度 * _beta + 重构误差 * (1 - _beta)

        Optimizer.zero_grad()  # 清空优化器的梯度
        loss.backward()  # 反向传播，计算梯度
        Optimizer.step()  # 更新模型参数

        print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))  # 打印当前 epoch 的编号、损失值和训练时间

    print("------------------------------------")  # 打印分隔线，表示训练结束
# def train2(A,ll):
#     batch_size = 128
#     num_epochs = 500
#     expect_tho = 0.05
#     Arm=torch.FloatTensor(A) #(39,662)
#     Arm = Arm.unsqueeze(0) #(1,39,662) 增加维度
#     Arm = Arm.unsqueeze(1)  #(1,1,39,662) 增加维度
#     emb = np.size(A, axis=1) # 662
#     m = np.size(A,axis=0)    # 39
#
#     def KL_devergence(p, q):
#         """
#         Calculate the KL-divergence of (p,q)
#         :param p:
#         :param q:
#         :return:
#         """
#         q = torch.nn.functional.softmax(q, dim=0)
#         q = torch.sum(q,
#                       dim=0) / batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
#         s1 = torch.sum(p * torch.log(p / q))
#         s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
#         return s1 + s2
#
#     model = AutoEncoder(emb=emb)
#     # if torch.cuda.is_available():
#     #model.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上
#
#
#     # 定义期望平均激活值和KL散度的权重
#     Optimizer = optim.Adam(model.parameters(), lr=0.1)
#     tho_tensor = torch.FloatTensor([expect_tho for _ in range(32)])  #32个 0.05  #64,32,128
#     # if torch.cuda.is_available():
#     #   tho_tensor = tho_tensor.cuda()
#     _beta = 0.1
#
# #    x = A
# #    modelN = ModelVAE(x=x, h_dim=128, z_dim=32, distribution='normal')
# #    optimizerS = OptimizerVAE(modelN)
#
#     for epoch in range(num_epochs): # 500轮
#         time_epoch_start = time.time()
#         l, decoder_out = model(Arm)    #
#
#         #        session.run(optimizerS.train_step)
#
#  #       encoder_out= modelN.z_var
#   #      session = tf.compat.v1.Session()
#    #     with session.as_default():
#     #        attr_embedding = encoder_out.eval()
#
#         attr_embedding = l.detach().numpy()
#
#
#
#         if ll==1:
#             np.savetxt("Data/attr embedding_m.txt", attr_embedding)
#
#         if ll==0:
#             np.savetxt("Data/attr embedding_d.txt", attr_embedding)
#
#         if ll==2:
#             np.savetxt("Data/Sd_dis embedding.txt",attr_embedding)
#         if ll==3:
#             np.savetxt("Data/Sm_dis embedding.txt",attr_embedding)
#
#         loss = F.mse_loss(decoder_out, Arm)
#         # 计算并增加KL散度到loss
#         _kl = KL_devergence(tho_tensor, l)
#         loss = _beta * _kl+loss*(1-_beta)
#
#         Optimizer.zero_grad()
#         loss.backward()
#         Optimizer.step()
#         print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))
#         #print("------------------------------------")
#
#
#
#     print("------------------------------------")

class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()


    def forward(self, x): # (1,6,39,662)
        concatenated_tensor = torch.cat([x[:, i, :, :] for i in range(x.shape[1])], dim=2)
        concatenated_tensor = torch.cat([concatenated_tensor[i, :, :] for i in range(x.shape[0])], dim=0)
        return concatenated_tensor  # (39, 6*662) =(39,3972)

class unReshapeLayer(nn.Module):
    def __init__(self):
        super(unReshapeLayer, self).__init__()

    def forward(self, x):
        height, width = x.shape
        channels = 6
        #channels = 9
        split_tensors = torch.split(x, width // channels, dim=1)
        #print(len(split_tensors))

        # 将每个分割后的张量在新的维度上扩展，变成 (batch_size, channels, height, width) 的大小
        batch_size = 1
        expanded_tensors = [split_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1) for split_tensor in split_tensors]
        #expanded_tensors = torch.tensor(expanded_tensors)

        # 拼接多个张量恢复成原始形状
        reconstructed_tensor = torch.cat(expanded_tensors, dim=1)

        return reconstructed_tensor # (1,6,39,662)  回到原来的形状了


#---------------------- 主体 AutoEncoder 模型 ----------------------
class AutoEncoder(nn.Module):
    def __init__(self, emb): #emb=662
        super(AutoEncoder,self).__init__()

        # 编码
        self.encoder = nn.Sequential(
            ResidualConvBlock(1, 3),  # 输入: (1, 1, 39, 662) → 输出: (1, 3, 39, 662)
            ResidualConvBlock(3, 6),  # 输出: (1, 6, 39, 662)
            # ResidualConvBlock(6, 9),
            nn.ReLU()
        )



        self.reshape = ReshapeLayer()  # → (39, 3972)
        #self.fc1 = nn.Linear(emb * 6, 128)

        self.fc1 = nn.Linear(emb * 6, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)     # 这个是输出维度 32

        # 解码器
        # self.dfc1 = nn.Linear(32, 64)
        # self.dfc2 = nn.Linear(64, 128)
        # self.dfc3 = nn.Linear(128, emb * 6)

        self.dfc1 = nn.Linear(32, 64)
        self.dfc2 = nn.Linear(64, 128)
        # self.dfc3 = nn.Linear(128, 256)
        #self.dfc3 = nn.Linear(128, emb * 6)
        self.dfc3 = nn.Linear(128, emb * 6)
        self.unreshape = unReshapeLayer()  # → (1, 6, 39, 662)

        self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=9, out_channels=6, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU()
        )

    def forward(self, x):  # x: (1, 1, 39, 662)
        x = self.encoder(x)             # → (1, 6, 39, 662)
        x = self.reshape(x)             # → (39, 3972)

        # 全连接 + LayerNorm
        x = F.relu(self.ln1(self.fc1(x)))  # → (39, 128)
        x = F.relu(self.ln2(self.fc2(x)))  # → (39, 64)
        x = F.relu(self.ln3(self.fc3(x)))  # → (39, 32)
        # x = F.relu(self.ln4(self.fc4(x)))  # → (39, 32)
        encoder_out = x

        # 解码器
        x = F.relu(self.dfc1(x))  # → (39, 64)
        x = F.relu(self.dfc2(x))  # → (39, 128)
        x = self.dfc3(x)  # → (39, 3972)

     #    x = F.relu(self.dfc1(x))          # → (39, 64)
     #    x = F.relu(self.dfc2(x))          # → (39, 128)
     # #   x = F.relu(self.dfc3(x))
     #    x = self.dfc4(x)                  # → (39, 3972)
        x = self.unreshape(x)             # → (1, 6, 39, 662)
        decoder_out = self.decoder_conv(x)  # → (1, 1, 39, 662)

        return encoder_out, decoder_out


# class AutoEncoder(nn.Module):
#     def __init__(self,emb): #emb=662
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=3,kernel_size=3,padding=1),
#            # nn.BatchNorm2d(3),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,padding=1),
#             nn.ReLU(),
#             ReshapeLayer(),
#             nn.Linear(emb*6,128),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64,32),
#             nn.ReLU()
#                     )
#
#
#         self.decoder = nn.Sequential(
#             nn.Linear(32,64),
#             nn.ReLU(),
#             nn.Linear(64,128),
#             nn.ReLU(),
#             nn.Linear(128,emb*6),
#             nn.ReLU(),
#             unReshapeLayer(),
#             nn.ConvTranspose2d(in_channels=6,out_channels=3,kernel_size=3,padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=3,out_channels=1,kernel_size=3,padding=1),
#             nn.ReLU(),
#
#         )
#
#     def forward(self, x):
#         encoder_out = self.encoder(x)
#         decoder_out = self.decoder(encoder_out)
#
#
#         return encoder_out, decoder_out

# ########
# #
# #
# # class AutoEncoder(nn.Module):
# #     def __init__(self, emb):  # 初始化函数，emb 是输入数据的特征维度，emb=662
# #         super(AutoEncoder, self).__init__()  # 调用父类 nn.Module 的初始化函数
# #
# #         # 定义编码器部分
# #         self.encoder = nn.Sequential(     # (1, 1, 39, 662)
# #             ResidualConvBlock(1, 3),  # 第一个残差卷积块，输入1通道，输出3通道
# #             ResidualConvBlock(3, 6),  # 第二个残差卷积块，输入3通道，输出6通道
# #            #  nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),  # 第一层卷积，输入通道 1，输出通道 3，卷积核大小 3x3，填充 1 第一层卷积 输入尺寸为 (1, 1, 39, 662) 输出尺寸为 (1, 3(通道数), 39, 662)
# #            #  # nn.BatchNorm2d(3),  # 批归一化层（注释掉了）
# #            #  nn.ReLU(),  # ReLU 激活函数 不改变形状(1,3,39,662)
# #            #  nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),  # 第二层卷积，输入通道 3，输出通道 6，卷积核大小 3x3，填充 1  输出形状：(1, 6, 39, 662)  out_channels = 卷积核数量
# #            # # nn.GroupNorm(2, 6),  # 或nn.LayerNorm
# #            #  nn.ReLU(),  # ReLU 激活函数 不改变形状(1,6,39,662)
# #             ReshapeLayer(),  # 自定义层，用于调整张量形状   输入(1, 6, 39, 662) 输出形状：(39, 6 * 662) = (39, 3972)  textflatten 是一个自定义函数，它的功能可能是将输入的文本数据进行某种扁平化处理
# #             nn.Linear(emb * 6, 128),  # 全连接层，输入维度 emb*6，输出维度 128 emb=662,  输入形状：(39, 3972) 输出形状：(39, 128)
# #             nn.ReLU(),  # ReLU 激活函数
# #             nn.Linear(128, 64),  # 全连接层，输入维度 128，输出维度 64  输出形状：(39, 64)
# #             nn.ReLU(),  # ReLU 激活函数
# #             nn.Linear(64, 32),  # 全连接层，输入维度 64，输出维度 32   输出形状：(39, 32)
# #             nn.ReLU()  # ReLU 激活函数
# #         )
# #
# #         # 定义解码器部分
# #         self.decoder = nn.Sequential(
# #             nn.Linear(32, 64),  # 全连接层，输入维度 32，输出维度 64  输入形状：(39, 32) 输出形状：(39, 64)
# #             nn.ReLU(),  # ReLU 激活函数
# #             nn.Linear(64, 128),  # 全连接层，输入维度 64，输出维度 128 输出形状：(39, 128)
# #             nn.ReLU(),  # ReLU 激活函数
# #             nn.Linear(128, emb * 6),  # 全连接层，输入维度 128，输出维度 emb*6  输出形状：(39, 3972)
# #             nn.ReLU(),  # ReLU 激活函数
# #             unReshapeLayer(),  # 自定义层，用于恢复张量形状  输出形状：(1, 6, 39, 662)
# #             nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),  # 反卷积层，输入通道 6，输出通道 3，卷积核大小 3x3，填充 1  输出形状 (1, 3, 39, 662)
# #             nn.ReLU(),  # ReLU 激活函数
# #             nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),  # 反卷积层，输入通道 3，输出通道 1，卷积核大小 3x3，填充 1   输出形状：(1, 1, 39, 662)
# #             nn.ReLU()  # ReLU 激活函数
# #         )
# #
#     def forward(self, x):  # 前向传播函数 x是 (1, 1, 39, 662)
#         encoder_out = self.encoder(x)  # 将输入 x 传入编码器，得到编码器的输出 encoder_out
#         decoder_out = self.decoder(encoder_out)  # 将编码器的输出 encoder_out 传入解码器，得到解码器的输出 decoder_out
#         return encoder_out, decoder_out  # 返回编码器的输出和解码器的输出    decoder_out 拿去和原来的Arm计算损失函数，encoder_out 回复到原来的维度,拿去到矩阵融合中了
#     #encoder_out 是编码器的输出，形状为 (39, 32)  decoder_out 是解码器的输出，形状为 (1, 1, 39, 662)，与输入数据 x 的形状一致。



# ---------------------- 卷积残差块 ----------------------
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):  # 初始化函数，in_channels =1是输入通道数，out_channels 是输出通道数 =3
        super().__init__()  # 调用父类 nn.Module 的初始化方法

        # 判断输入通道数是否和输出通道数相同，用于决定是否需要投影匹配维度
        self.same_shape = (in_channels == out_channels)

        # 定义主分支的卷积操作，使用 3x3 卷积核 + padding=1 保证空间尺寸不变
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 定义 ReLU 激活函数（可以复用）
        self.relu = nn.ReLU()

        # 定义残差分支（shortcut）：
        # 如果输入和输出通道数不一致，需要通过 1x1 卷积调整维度使其可以相加
        if not self.same_shape:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            # 如果通道一致，残差部分直接返回原始输入（Identity）
            self.residual_conv = nn.Identity()

    def forward(self, x):
        """
        输入：
            x: 输入特征图，形状为 (B, C_in, H, W)
        输出：
            输出特征图，形状为 (B, C_out, H, W)，经过残差连接后再 ReLU
        """
        # 主分支卷积（对输入做特征提取）
        out = self.conv(x)

        # 残差路径：如果维度不一致会被映射成一致维度
        res = self.residual_conv(x)

        # 两条路径相加（残差连接），然后通过激活函数
        return self.relu(out + res)


#
# class AutoEncoder(nn.Module):
#     def __init__(self, emb):  # 初始化函数，emb 是输入数据的特征维度，emb=662
#         super(AutoEncoder, self).__init__()  # 调用父类 nn.Module 的初始化函数
#
#         # 定义编码器部分
#         self.encoder = nn.Sequential(     # (1, 1, 39, 662)
#             nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),  # 第一层卷积，输入通道 1，输出通道 3，卷积核大小 3x3，填充 1 第一层卷积 输入尺寸为 (1, 1, 39, 662) 输出尺寸为 (1, 3(通道数), 39, 662)
#             # nn.BatchNorm2d(3),  # 批归一化层（注释掉了）
#             nn.ReLU(),  # ReLU 激活函数 不改变形状(1,3,39,662)
#             nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),  # 第二层卷积，输入通道 3，输出通道 6，卷积核大小 3x3，填充 1  输出形状：(1, 6, 39, 662)  out_channels = 卷积核数量
#             nn.ReLU(),  # ReLU 激活函数 不改变形状(1,6,39,662)
#             ReshapeLayer(),  # 自定义层，用于调整张量形状   输入(1, 6, 39, 662) 输出形状：(39, 6 * 662) = (39, 3972)  textflatten 是一个自定义函数，它的功能可能是将输入的文本数据进行某种扁平化处理
#             nn.Linear(emb * 6, 128),  # 全连接层，输入维度 emb*6，输出维度 128 emb=662,  输入形状：(39, 3972) 输出形状：(39, 128)
#             nn.ReLU(),  # ReLU 激活函数
#             nn.Linear(128, 64),  # 全连接层，输入维度 128，输出维度 64  输出形状：(39, 64)
#             nn.ReLU(),  # ReLU 激活函数
#             nn.Linear(64, 32),  # 全连接层，输入维度 64，输出维度 32   输出形状：(39, 32)
#             nn.ReLU()  # ReLU 激活函数
#         )
#
#         # 定义解码器部分
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 64),  # 全连接层，输入维度 32，输出维度 64  输入形状：(39, 32) 输出形状：(39, 64)
#             nn.ReLU(),  # ReLU 激活函数
#             nn.Linear(64, 128),  # 全连接层，输入维度 64，输出维度 128 输出形状：(39, 128)
#             nn.ReLU(),  # ReLU 激活函数
#             nn.Linear(128, emb * 6),  # 全连接层，输入维度 128，输出维度 emb*6  输出形状：(39, 3972)
#             nn.ReLU(),  # ReLU 激活函数
#             unReshapeLayer(),  # 自定义层，用于恢复张量形状  输出形状：(1, 6, 39, 662)
#             nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),  # 反卷积层，输入通道 6，输出通道 3，卷积核大小 3x3，填充 1  输出形状 (1, 3, 39, 662)
#             nn.ReLU(),  # ReLU 激活函数
#             nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),  # 反卷积层，输入通道 3，输出通道 1，卷积核大小 3x3，填充 1   输出形状：(1, 1, 39, 662)
#             nn.ReLU()  # ReLU 激活函数
#         )

#
# class EarlyStopping:
#     def __init__(self, patience=5, delta=1e-4):
#         self.patience = patience  # 容忍轮数
#         self.delta = delta        # 最小变化幅度
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False
#
#     def __call__(self, current_loss):
#         if self.best_loss is None:
#             self.best_loss = current_loss
#         elif current_loss > self.best_loss - self.delta:
#             self.counter += 1
#             print(f"早停计数：{self.counter}/{self.patience}")
#             if self.counter >= self.patience:
#                 print(">>> 触发早停机制！")
#                 self.early_stop = True
#         else:
#             self.best_loss = current_loss
#             self.counter = 0
#
#
# def train2(A, ll):
#     batch_size = 128
#     num_epochs = 50  # 设置最大轮数，可提前终止
#     expect_tho = 0.05
#
#     Arm = torch.FloatTensor(A).unsqueeze(0).unsqueeze(1)
#     emb = np.size(A, axis=1)
#     m = np.size(A, axis=0)
#
#     def KL_devergence(p, q):
#         q = torch.nn.functional.softmax(q, dim=0)
#         q = torch.sum(q, dim=0) / batch_size
#         s1 = torch.sum(p * torch.log(p / q))
#         s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
#         return s1 + s2
#
#     model = AutoEncoder(emb=emb)
#     Optimizer = optim.Adam(model.parameters(), lr=0.1)
#
#     tho_tensor = torch.FloatTensor([expect_tho for _ in range(32)])
#     _beta = 0.1
#
#     # ✅ 创建 EarlyStopping 实例
#     early_stopper = EarlyStopping(patience=6, delta=1e-4)
#
#     for epoch in range(num_epochs):
#         time_epoch_start = time.time()
#         l, decoder_out = model(Arm)
#         attr_embedding = l.detach().numpy()
#
#         # 保存嵌入向量
#         if ll == 1:
#             np.savetxt("Data1/attr embedding_m.txt", attr_embedding)
#         elif ll == 0:
#             np.savetxt("Data1/attr embedding_d.txt", attr_embedding)
#
#         # 计算总损失
#         loss = F.mse_loss(decoder_out, Arm)
#         _kl = KL_devergence(tho_tensor, l)
#         loss = _beta * _kl + loss * (1 - _beta)
#
#         Optimizer.zero_grad()
#         loss.backward()
#         Optimizer.step()
#
#         print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss.item(), time.time() - time_epoch_start))
#
#         # ✅ 执行早停判断
#         early_stopper(loss.item())
#         if early_stopper.early_stop:
#             print(">>> 提前停止训练")
#             break
#
#     print("--------训练结束--------")
