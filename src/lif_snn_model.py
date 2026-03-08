import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (1 + input.abs())**2
        return grad


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):

    def __init__(self, input_size, output_size, tau=0.9, threshold=1.0):

        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

        self.tau = tau
        self.threshold = threshold

    def forward(self, x):

        mem = torch.zeros(x.size(0), self.fc.out_features)

        mem = self.tau * mem + self.fc(x)

        spike = spike_fn(mem - self.threshold)

        mem = mem * (1 - spike)

        return spike


class GestureSNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.lif1 = LIFLayer(16, 32)

        self.lif2 = LIFLayer(32, 32)

        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):

        x = self.lif1(x)

        x = self.lif2(x)

        out = torch.sigmoid(self.fc_out(x))

        return out