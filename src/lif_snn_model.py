import torch
import torch.nn as nn


class LIFLayer(nn.Module):

    def __init__(self, input_size, output_size, tau=0.9, threshold=1.0):

        super().__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):

        batch_size, time_steps, input_size = x.shape

        mem = torch.zeros(batch_size, self.fc.out_features)
        spikes = []

        for t in range(time_steps):

            input_t = x[:, t, :]

            mem = self.tau * mem + self.fc(input_t)

            spike = (mem > self.threshold).float()

            mem = mem * (1 - spike)

            spikes.append(spike)

        spikes = torch.stack(spikes, dim=1)

        return spikes


class GestureSNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.lif1 = LIFLayer(8, 16)
        self.lif2 = LIFLayer(16, 16)

        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):

        x = self.lif1(x)

        x = self.lif2(x)

        spike_rate = x.mean(dim=1)

        out = torch.sigmoid(self.fc_out(spike_rate))

        return out