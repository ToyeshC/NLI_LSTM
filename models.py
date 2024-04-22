import torch
import torch.nn as nn

class Model_Loader(nn.Module):
    def __init__(self, config):
        super(Model_Loader, self).__init__()

        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.encoder_dim = config['encoder_dim']
        self.encoder_type = config['encoder_type']
        self.embedding_dim = config['embedding_dim']
        self.encoder = eval(self.encoder_type)(config)
        self.input_dim = 0

        if self.encoder_type == "BasicEncoder":
            self.input_dim = self.embedding_dim * 4
        elif self.encoder_type == "Uni_LSTM":
            self.input_dim = 4*self.encoder_dim
        elif self.encoder_type == "Bi_LSTM" or self.encoder_type == "Bi_LSTM_Max":
            self.input_dim = 4*2*self.encoder_dim
 
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Linear(self.fc_dim, self.classes)
            )

    def forward(self, hypothesis, premise):
        h = self.encoder(hypothesis)
        p = self.encoder(premise)

        features = torch.cat((h, p, torch.abs(h-p), h*p), 1)
        output = self.classifier(features)
        return output

    def encode(self, hypothesis):
        embedding = self.encoder(hypothesis)
        return embedding


class Basic_Encoder(nn.Module):
    def __init__(self, config):
        super(Basic_Encoder, self).__init__()
        self.embedding_dim = config['embedding_dim']

    def forward(self, embedding):
        input_batch, _ = embedding
        result = torch.mean(input_batch, dim=1)

        return result



class Uni_LSTM(nn.Module):
    def __init__(self, config):
        super(Uni_LSTM, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.encoder_dim = config['encoder_dim']
        self.model_new = config['model_new']
        self.batch_size = config['batch_size']
        self.lstm_encoder = nn.LSTM(self.embedding_dim, self.encoder_dim, dropout=self.model_new)

    def forward(self, sentence_list):
        sentence, sentence_length = sentence_list
        packed = nn.utils.rnn.pack_padded_sequence(sentence, sentence_length, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm_encoder(packed)
        result = h_n[0, ...]
        
        return result


class Bi_LSTM(nn.Module):
    def __init__(self, config):
        super(Bi_LSTM, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.encoder_dim = config['encoder_dim']
        self.model_new = config['model_new']
        self.lstm_encoder = nn.LSTM(self.embedding_dim, self.encoder_dim, dropout=self.model_new, bidirectional=True)
        
    def forward(self, sentence_list):
        sentence, sentence_length = sentence_list
        packed = nn.utils.rnn.pack_padded_sequence(sentence, sentence_length, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm_encoder(packed)
        result = torch.cat((h_n[0], h_n[1]), dim=-1)

        return result


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
