import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class DatasetLoader(Dataset):
    def __init__(self, dataset, word_to_vec, embedding_dim=300):
        self.dataset = dataset
        self.word_to_vec = word_to_vec
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.data['premise'])

    def __getitem__(self, idx):
        premise = self.data['premise'][idx]
        hypothesis = self.data['hypothesis'][idx]
        label = self.data['label'][idx]
        premise_index, premise_length = self.sentence_indexer(premise)
        hypothesis_index, hypothesis_length = self.sentence_indexer(hypothesis)

        return premise_index, premise_length, hypothesis_index, hypothesis_length, label

    def sentence_indexer(self, sentence):
        tokens = sentence.split()
        index = torch.tensor([self.word_to_vec.get(word, [0]*self.embedding_dim) for word in tokens], dtype=torch.float32)

        length = len(index)

        return index, length


def get_vocab(sentences):
    # create vocab of words
    vocabulary_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in vocabulary_dict:
                vocabulary_dict[word] = ''
    vocabulary_dict['<s>'] = ''
    vocabulary_dict['</s>'] = ''
    vocabulary_dict['<p>'] = ''
    return vocabulary_dict


def get_glove_embeddings(vocabulary_dict, glove_path):
    word_to_vec = {}

    with open(glove_path, encoding='utf-8') as file:
        for line in file:
            word, vec = line.split(' ', 1)
            if word in vocabulary_dict:
                word_to_vec[word] = list(map(float, vec.split()))

    return word_to_vec


def build_vocab(sentences, glove_path):
    vocabulary_dict = get_vocab(sentences)
    word_to_vec = get_glove_embeddings(vocabulary_dict, glove_path)
    return word_to_vec


def collate_fn(batch):
    premise, s1_lengths, hypothesis, s2_lengths, labels = zip(*batch)
    
    premise = torch.nn.utils.rnn.pad_sequence(premise, batch_first=True, padding_value=0)
    hypothesis = torch.nn.utils.rnn.pad_sequence(hypothesis, batch_first=True, padding_value=0)
    s1_lengths = torch.tensor(s1_lengths)
    s2_lengths = torch.tensor(s2_lengths)
    labels = torch.tensor(labels, dtype=torch.long)

    return premise, s1_lengths, hypothesis, s2_lengths, labels
