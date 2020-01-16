import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCrossEntropyLossContinuous(nn.CrossEntropyLoss):
    def __init__(self, bandwidth, *args):
        self.bandwidth = bandwidth
        super().__init__(*args)

    def forward(self, input_logits, target_logits):
        assert len(input_logits.size()) == 2, 'wrong size for input logits'
        assert len(target_logits.size()) == 2, 'wrong size for target logits'

        a = F.softmax(target_logits, 1)
        a_log = F.log_softmax(target_logits, 1)
        b = F.log_softmax(input_logits, 1)

        xent = - (a * b).sum(1)
        ent = - (a * a_log).sum(1)  # Entropy of targets
        rbf_xent = torch.exp(-((xent - ent) ** 2) / (2 * self.bandwidth ** 2))

        return torch.mean(rbf_xent)
