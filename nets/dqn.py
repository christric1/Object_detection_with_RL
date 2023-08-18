import torch.nn as nn


class dqn(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        '''        
            Input  : in_dim -> image feature + history vector
            Output : out_dim
        '''
        super(dqn, self).__init__()
        self.classifier = nn.Sequential(
            # 全連接層
            nn.Linear(in_dim, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(4096, out_dim),
        )

    def forward(self, x):
        return self.classifier(x)