# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import time

if __name__=='__main__':
    config = Config()
    train_file = '../data/data/job_dataset_converted_train.json'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = '../data/data/job_dataset_converted_test.json'
    valid_file ='../data/data/job_dataset_converted_valid.json'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    w2v_file = 'fasttext.en.300d'
    
    dataset = Dataset(config)
    dataset.load_data(w2v_file, train_file, test_file,valid_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = TextRNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    BCELog = nn.BCEWithLogitsLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(BCELog)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    startTime = time.time()
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
    executionTime = (time.time() - startTime)
    print("Training time",executionTime)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)
    print ('Final Training Accuracy:',train_acc)
    print ('Final Validation Accuracy:',val_acc)
    print ('Final Test Accuracy:',test_acc)
