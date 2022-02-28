import torch.nn as nn
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .layers import get_output_layers

CHECKPOINTS = {"bert": "bert-base-uncased"}


class Teacher(nn.Module):
    def __init__(self, model_name, label_map):
        super(Teacher, self).__init__()

        self.model_name = model_name
        checkpoint = CHECKPOINTS.get(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(checkpoint, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(checkpoint, config=self.config)

        self.output_layers = get_output_layers(self.config.hidden_size, label_map)

    def forward(self, x):
        sequence_outs, pooled_out, _ = eval(
            f"self.encoder.{self.encoder.base_model_prefix}(x)"
        )
        outputs = {
            classname: layer(sequence_outs) if classname == "ner" else layer(pooled_out)
            for classname, layer in self.output_layers.items()
        }

        return outputs

    def get_word_embeddings(self, word_emb_dim):
        embedding_matrix = eval(
            f"self.encoder.{self.encoder.base_model_prefix}.embeddings.weight.numpy()"
        )

        if embedding_matrix.shape[1] > word_emb_dim:
            pca = PCA(n_components=word_emb_dim)
            downscaled_embeddings = pca.fit_transform(embedding_matrix)
            embeddings = dict(zip(self.tokenizer.get_vocab(), downscaled_embeddings))

        return embeddings

    def get_special_tokens(self):
        tokens = {}
        if self.model_name in ["bert", "muril"]:
            tokens["bos"] = self.tokenizer.cls_token
            tokens["eos"] = self.tokenizer.sep_token
            tokens["pad"] = self.tokenizer.pad_token
        else:
            tokens["bos"] = "<s>"
            tokens["eos"] = "</s>"
            tokens["pad"] = "<pad>"

        return tokens
