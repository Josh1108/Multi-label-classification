# utils.py

import torch
from torchtext import data
from torchtext.vocab import Vectors
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import json
import recsys_metrics
import time

def metric_calc(true_labels, predicted_labels):
    dicti={}
    for k in [5,10,20,50,100]:
        dicti[f'Precision@{k}'] = recsys_metrics.precision(predicted_labels,true_labels,k)
        dicti[f'Recall@{k}'] = recsys_metrics.recall(predicted_labels,true_labels,k)
        dicti[f'ndcg@k'] = recsys_metrics.normalized_dcg(predicted_labels,true_labels,k)

    dicti['mrr'] = recsys_metrics.mean_reciprocal_rank(predicted_labels,target=true_labels)
    return dicti

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, labels):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return [x for x in labels]

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        data_text =[]
        data_label = []
        with open(filename, 'r') as datafile:
            data = [line.strip() for line in datafile]
            for row in data:
                dicti = json.loads(row)
                _data_text = " ".join(dicti["features_content"])
                data_text.append(_data_text)
                _data_label = dicti["labels_index"]
                lbls=[]
                for i in range(self.config.output_size):
                    if i in _data_label:
                        lbls.append(1)
                    else:
                        lbls.append(0)
                data_label.append(lbls)
        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data, vectors=w2v_file)
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    starttime = time.time()
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        all_preds.extend(y_pred)
        all_y.extend(batch.label)
    endtime = time.time() - starttime
    print("time taken in ms for pred", endtime/(len(all_preds)*1000))
    score = metric_calc(torch.stack(all_y).cpu(), torch.stack(all_preds).cpu())
    return score