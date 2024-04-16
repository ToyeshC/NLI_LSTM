class uni_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(uni_LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(glove_vectors.vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, premise, hypothesis):
        premise_embedded = self.embedding(premise)
        _, (premise_hidden, _) = self.lstm(premise_embedded)
        premise_hidden = premise_hidden.squeeze(0)
        
        hypothesis_embedded = self.embedding(hypothesis)
        _, (hypothesis_hidden, _) = self.lstm(hypothesis_embedded)
        hypothesis_hidden = hypothesis_hidden.squeeze(0)

        concatenated = torch.cat((premise_hidden, hypothesis_hidden), dim=1)
        output = self.fc(concatenated)
        return output