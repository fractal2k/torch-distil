import torch.nn as nn


def get_output_layers(in_features, label_map):
    layers = {}
    for classname, values in label_map.items():
        layers[classname] = nn.Linear(in_features=in_features, out_features=len(values))

    return layers


def get_embedding_layer(embeddings, padding_idx):
    return nn.Embedding.from_pretrained(
        embeddings, freeze=False, padding_idx=padding_idx
    )


def get_bilstm_layer(input_size, hidden_size, proj_size=0):
    return nn.LSTM(
        input_size,
        hidden_size,
        batch_first=True,
        bidirectional=True,
        proj_size=proj_size,
    )
