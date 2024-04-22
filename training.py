import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from data_loader import DataLoader, get_nli, vocab_loader, collate_fn
from models import Model_Loader


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='data/', help="SNLI data path")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="glove/glove.840B.300d.txt", help="GLoVe word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")

# model
parser.add_argument("--encoder_type", type=str, default='BasicEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")

# gpu
parser.add_argument("--seed", type=int, default=42, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

print("Number of words in word_vec:", len(word_vec))
# print("Sample word_vec items:", list(word_vec.items())[:2])


train_dataset = Daa(train, word_vec)
valid_dataset = NLIDataset(valid, word_vec)
test_dataset = NLIDataset(test, word_vec)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)         ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'dpout_model'    :  params.dpout_model    ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'vector_embeddings' : word_vec            ,
    }

# model
encoder_types = ['Basic_Encoder', 'Uni_LSTM', 'Bi_LSTM',
                 'Bi_LSTM_Max']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLIClassifier(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn = optim.SGD
lr = params.lr
lrshrink = params.lrshrink
minlr = params.minlr
optimizer = optim_fn(nli_net.parameters(), lr=lr)

# cuda by default
nli_net.to(device)
loss_fn.to(device)

"""
TRAIN
"""
val_acc_best = -1e10
stop_training = False


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    print('Learning rate : {0}'.format(lr))

    for stidx, (s1_batch, s1_len, s2_batch, s2_len, tgt_batch) in tqdm(enumerate(train_loader)):
        s1_batch, s2_batch = s1_batch.to(device), s2_batch.to(device)
        tgt_batch = torch.LongTensor(tgt_batch).to(device)

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # optimizer step
        optimizer.step()

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; accuracy train : {3}'.format(
                stidx, round(np.mean(all_costs), 2),
                int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                round(100.*correct/((stidx+1)*params.batch_size), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []

    train_acc = round(100 * correct/len(train_loader.dataset), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc

def evaluate(epoch, dataloader, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    for s1_batch, s1_len, s2_batch, s2_len, tgt_batch in dataloader:
        s1_batch, s2_batch = s1_batch.to(device), s2_batch.to(device)
        tgt_batch = torch.LongTensor(tgt_batch).to(device)

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

    accuracy = round(100 * correct / len(dataloader.dataset), 2)
    print('results : epoch {0} ; mean accuracy {1} : {2}'.format(epoch, eval_type, accuracy))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if accuracy > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = accuracy
        else:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / lrshrink
            print('Shrinking lr by : {0}. New lr = {1}'
                .format(lrshrink,
                        optimizer.param_groups[0]['lr']))
            if optimizer.param_groups[0]['lr'] < minlr:
                stop_training = True


    # if eval_type == 'valid' and epoch <= params.n_epochs:
    #     if accuracy > val_acc_best:
    #         print('saving model at epoch {0}'.format(epoch))
    #         torch.save(nli_net.state_dict(), os.path.join(params.outputdir, params.outputmodelname))
    #         val_acc_best = accuracy
    return accuracy

# Train the model
epoch = 1
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, valid_loader)
    epoch += 1

# Run the model on the test set
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
print("Testing the model...")
test_acc = evaluate(1, test_loader, 'test', True)
print("Test accuracy: {}".format(test_acc))
