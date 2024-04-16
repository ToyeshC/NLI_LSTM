import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(BaselineModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding.from_pretrained(glove_vectors.vectors)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, premise, hypothesis):
        premise_embedded = self.embedding(premise).mean(dim=1)
        hypothesis_embedded = self.embedding(hypothesis).mean(dim=1)
        concatenated = torch.cat((premise_embedded, hypothesis_embedded), dim=1)
        output = self.fc(concatenated)
        return output