class max_bi_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(max_bi_LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(glove_vectors.vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, premise, hypothesis):
        premise_embedded = self.embedding(premise)
        lstm_out, _ = self.lstm(premise_embedded)
        premise_hidden = torch.max(lstm_out, dim=1)[0]
        
        hypothesis_embedded = self.embedding(hypothesis)
        lstm_out, _ = self.lstm(hypothesis_embedded)
        hypothesis_hidden = torch.max(lstm_out, dim=1)[0]

        concatenated = torch.cat((premise_hidden, hypothesis_hidden), dim=1)
        output = self.fc(concatenated)
        return output