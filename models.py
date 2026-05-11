import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden1=512, hidden2=256, dropout=0.3, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)