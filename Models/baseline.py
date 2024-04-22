import torch
import torch.nn as nn

class Basic_Encoder(nn.Module):
    def __init__(self, config):
        super(Basic_Encoder, self).__init__()
        self.embedding_dim = config['embedding_dim']

    def forward(self, embedding):
        input_batch, _ = embedding
        result = torch.mean(input_batch, dim=1)

        return result
