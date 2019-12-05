import torch

from .base import Embeddings


class BertEmbeddings(Embeddings):

    def __init__(self):
        super().__init__()
        self.tokenizer = torch.hub.load(
            'huggingface/pytorch-transformers',
            'tokenizer',
            'bert-base-multilingual-cased'
        )
        self.model = torch.hub.load(
            'huggingface/pytorch-transformers',
            'model',
            'bert-base-multilingual-cased'
        )

        self.embedding_size = 768

    def forward(self, sentence: str):
        tokens = torch.LongTensor([self.tokenizer.encode(sentence)])
        with torch.no_grad():
            embedding, _ = self.model(tokens)

        return embedding.view((-1, self.embedding_size))
