import torch
import torch.nn as nn

class Bi_LSTM_Max(nn.Module):
    def __init__(self, config):
        super(Bi_LSTM_Max, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.encoder_dim = config['encoder_dim']
        self.model_new = config['model_new']
        self.lstm_encoder = nn.LSTM(self.embedding_dim, self.encoder_dim, batch_first=True, dropout=self.model_new, bidirectional=True)

    def forward(self, sentence_list):
        sentence, sentence_length = sentence_list
        packed = nn.utils.rnn.pack_padded_sequence(sentence, sentence_length, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm_encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        result, _ = torch.max(output, dim=1)

        return result
