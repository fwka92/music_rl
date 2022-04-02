"""
Implement a PPO training on MusicEnvironment and MusicDataset using Pytorch and tianshou for music generation
with deep learning.
"""

import numpy as np
import tianshou
import torch
import torch.nn as nn
from tqdm import tqdm

from music_dataset import MusicDataset
# build a super complex neural network using attention blocks and residual blocks
from music_environment import MusicEnvironment


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv4 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv5 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv6 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv7 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv8 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv9 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv10 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv11 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv12 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv13 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv14 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv15 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv16 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv17 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv18 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv19 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.conv20 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.bn5 = nn.BatchNorm1d(out_channels)
        self.bn6 = nn.BatchNorm1d(out_channels)
        self.bn7 = nn.BatchNorm1d(out_channels)
        self.bn8 = nn.BatchNorm1d(out_channels)
        self.bn9 = nn.BatchNorm1d(out_channels)
        self.bn10 = nn.BatchNorm1d(out_channels)
        self.bn11 = nn.BatchNorm1d(out_channels)
        self.bn12 = nn.BatchNorm1d(out_channels)
        self.bn13 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=(3,), stride=(stride,), padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=(3,), stride=(1,), padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x) + x


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=(3,), stride=(1,), padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4)
        self.layer3 = self._make_layer(32, 64, 6)
        self.layer4 = self._make_layer(64, 128, 3)
        self.avgpool = nn.AvgPool1d(8)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, 1))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(16, 16, 3)
        self.layer2 = self._make_layer(16, 32, 4)
        self.layer3 = self._make_layer(32, 64, 6)
        self.layer4 = self._make_layer(64, 128, 3)
        self.avgpool = nn.AvgPool1d(8)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, 1))
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


dataset = MusicDataset('musescore')
dataset.load_data()
seq_len = 1024
env = MusicEnvironment(dataset, seq_length=seq_len)

train_envs = tianshou.env.DummyVectorEnv([lambda: MusicEnvironment(dataset, seq_length=seq_len) for _ in range(10)])
test_envs = tianshou.env.DummyVectorEnv([lambda: MusicEnvironment(dataset, seq_length=seq_len) for _ in range(100)])


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = tianshou.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_collector = tianshou.data.Collector(policy, train_envs, tianshou.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = tianshou.data.Collector(policy, test_envs, exploration_noise=True)

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger


writer = SummaryWriter()
logger = TensorboardLogger(writer)
save_every = 10

# pre-collect at least 5000 transitions with random action before training
train_collector.collect(n_step=500, random=True)

policy.set_eps(0.1)
for i in tqdm(range(int(1e6))):  # total step
    collect_result = train_collector.collect(n_step=10)

    # once if the collected episodes' mean returns reach the threshold,
    # or every 1000 steps, we test it on test_collector
    if collect_result['rews'].mean() >= -1 or i % 1000 == 0:
        policy.set_eps(0.05)
        result = test_collector.collect(n_step=100, render=1/35)

    if i % save_every == 0:
        torch.save(policy.state_dict(), 'dqn.pth')
        torch.save(policy.state_dict(), '/content/drive/MyDrive/dqn.pth')

    # train policy with a sampled batch data from buffer
    losses = policy.update(64, train_collector.buffer)
    writer.add_scalar('loss', losses['loss'], i)

    # log some statistics
    writer.add_scalar('reward', collect_result['rews'].mean(), i)


