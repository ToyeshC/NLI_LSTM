import torch
from models import Model_Classifier, Basic_Encoder, Uni_LSTM, Bi_LSTM, Bi_LSTM_Max
import sys
import numpy as np
import logging
import data_loader
import os

senteval_path = './SentEval-main/'
senteval_data = './SentEval/data/'
model_pickle_path = './savedir/'
glove_vector_file = "glove/glove.840B.300d.txt"

sys.path.insert(0, senteval_path)
import senteval

def model_loader(model_type, model_path):
    model = Model_Classifier()
    model.encoder = model_type()
    model.load_state_dict(torch.load(model_path))
    return model.encoder


def prepare(params, sentences):
    samples_as_strings = [' '.join(sentence) for sentence in sentences]
    params.word_to_id = data_loader.get_vocab(samples_as_strings)
    params.word_to_vec = data_loader.vocab_loader(samples_as_strings, glove_vector_file)
    params.vector_dim = 300
    
    model_type = params["model_type"]
    model_path = os.path.join(model_pickle_path, params["model_file"])
    params["model"] = model_loader(model_type, model_path)
    params["model"].eval()

    pass

def batcher(params, batch):
    model = params["model"]

    dataset_temp = data_loader.DatasetLoader([], [], [])

    embeddings = []

    for sentence in batch:
        sentence = ' '.join(sentence)
        hypothesis_index, hypothesis_length = dataset_temp._get_sentence_indices(sentence)
        hypothesis_index = hypothesis_index.unsqueeze(0)
        hypothesis_length = torch.tensor([hypothesis_length])
        with torch.no_grad():
            sent_embedding = model((hypothesis_index, hypothesis_length)).numpy()
        embeddings.append(sent_embedding)
    embeddings = np.vstack(embeddings)
    
    return embeddings

params_senteval = {'task_path': senteval_data, 'usepytorch': False, 'kfold': 10}
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

models = [
    {"model_type": Basic_Encoder, "model_file": "Basic_Encoder.pickle"},
    {"model_type": Uni_LSTM, "model_file": "Uni_LSTM.pickle"},
    {"model_type": Bi_LSTM, "model_file": "Bi_LSTM.pickle"},
    {"model_type": Bi_LSTM_Max, "model_file": "Bi_LSTM_Max.pickle"}
]

def main():
    for model in models:
        params_senteval.update(model)
        se = senteval.SE(params_senteval, batcher, prepare)

        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'STS14']
        results = se.eval(transfer_tasks)
        print(results)

if __name__ == "__main__":
    main()
