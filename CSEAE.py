import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np


def train2(A, ll):
    batch_size = 128
    num_epochs = 25
    expect_tho = 0.05

    Arm = torch.FloatTensor(A)
    Arm = Arm.unsqueeze(0)
    Arm = Arm.unsqueeze(1)

    emb = np.size(A, axis=1)
    m = np.size(A, axis=0)

    def KL_devergence(p, q):
        q = torch.nn.functional.softmax(q, dim=0)
        q = torch.sum(q, dim=0) / batch_size
        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        return s1 + s2

    model = AutoEncoder(emb=emb)
    Optimizer = optim.Adam(model.parameters(), lr=0.1)
    tho_tensor = torch.FloatTensor([expect_tho for _ in range(32)])
    _beta = 0.1

    for epoch in range(num_epochs):
        time_epoch_start = time.time()
        l, decoder_out = model(Arm)

        attr_embedding = l.detach().numpy()

        if ll == 1:
            np.savetxt("HMDAD/attr embedding_m.txt", attr_embedding)
        elif ll == 0:
            np.savetxt("HMDAD/attr embedding_d.txt", attr_embedding)

        loss = F.mse_loss(decoder_out, Arm)
        _kl = KL_devergence(tho_tensor, l)
        loss = _beta * _kl + loss * (1 - _beta)

        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))

    print("------------------------------------")


class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()

    def forward(self, x):
        concatenated_tensor = torch.cat([x[:, i, :, :] for i in range(x.shape[1])], dim=2)
        concatenated_tensor = torch.cat([concatenated_tensor[i, :, :] for i in range(x.shape[0])], dim=0)
        return concatenated_tensor


class unReshapeLayer(nn.Module):
    def __init__(self):
        super(unReshapeLayer, self).__init__()

    def forward(self, x):
        height, width = x.shape
        channels = 6
        split_tensors = torch.split(x, width // channels, dim=1)
        batch_size = 1
        expanded_tensors = [split_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1) for split_tensor in split_tensors]
        reconstructed_tensor = torch.cat(expanded_tensors, dim=1)
        return reconstructed_tensor


class AutoEncoder(nn.Module):
    def __init__(self, emb):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ResidualConvBlock(1, 3),
            ResidualConvBlock(3, 6),
            nn.ReLU()
        )

        self.reshape = ReshapeLayer()
        self.fc1 = nn.Linear(emb * 6, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)

        self.dfc1 = nn.Linear(32, 64)
        self.dfc2 = nn.Linear(64, 128)
        self.dfc3 = nn.Linear(128, emb * 6)
        self.unreshape = unReshapeLayer()

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.reshape(x)

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        encoder_out = x

        x = F.relu(self.dfc1(x))
        x = F.relu(self.dfc2(x))
        x = self.dfc3(x)

        x = self.unreshape(x)
        decoder_out = self.decoder_conv(x)

        return encoder_out, decoder_out


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.same_shape = (in_channels == out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        if not self.same_shape:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        res = self.residual_conv(x)
        return self.relu(out + res)
