# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 64
    bidirectional = True
    output_size = 2548
    max_epochs = 30
    lr = 0.0003
    batch_size = 64
    max_sen_len = 60 # Sequence length for RNN
    dropout_keep = 0.8
    gru = 1