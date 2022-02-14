import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
          nn.Linear(in_features=input_dim, out_features=128,bias=True),
          nn.ReLU(),
          nn.Linear(in_features=128, out_features=out_dim,bias=True)
        )

    def forward(self, x):
      return self.mlp(x)