import torch
import torch.nn as nn

from .layers import get_embedding_layer, get_bilstm_layer, get_output_layers


class BiLSTM(nn.Module):
    def __init__(self, embeddings, lstm_hidden_size, padding_idx, label_map):
        super(BiLSTM, self).__init__()

        self.embedding_layer = get_embedding_layer(embeddings, padding_idx)
        emb_size = len(embeddings[0])
        self.bilstm_layer = get_bilstm_layer(emb_size, lstm_hidden_size)
        self.output_layers = get_output_layers(lstm_hidden_size * 2, label_map)

    def forward(self, x):
        # TODO: Figure out how to handle gradual unfreezing in the same module itself.
        pass
