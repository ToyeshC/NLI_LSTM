# importing libraries
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from torchtext.vocab import GloVe

# load dataset from hugging face
dataset = load_dataset('https://huggingface.co/datasets/stanfordnlp/snli')

# access the train, validation, and test splits
# train_data = dataset['train']
# val_data = dataset['validation']
# test_data = dataset['test']

# pre process the data set (tokenisation and lowercase)
def preprocess(text):
    # Tokenize the text using nltk word_tokenize
    tokens = word_tokenize(text)
    # Convert tokens to lowercase
    tokens_lower = [token.lower() for token in tokens]
    # Join tokens back into a single string
    processed_text = ' '.join(tokens_lower)
    return processed_text

# for split in ['train', 'validation', 'test']:
for split in ['validation']:
    for i in range(len(dataset[split])):
        premise = dataset[split][i]['premise']
        hypothesis = dataset[split][i]['hypothesis']
        dataset[split][i]['premise'] = preprocess(premise)
        dataset[split][i]['hypothesis'] = preprocess(hypothesis)


# load the GloVe embeddings
glove_vectors = GloVe(name='840B', dim=300)