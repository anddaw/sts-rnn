from .base import Embeddings


class CachedEmbeddings(Embeddings):

    def __init__(self, embeddings: Embeddings):
        super().__init__()
        self.wrapped = embeddings
        self.embedding_size = embeddings.embedding_size
        self.cache = {}

    def forward(self, sentence: str):

        if sentence in self.cache:
            return self.cache[sentence]
        else:
            embedding = self.wrapped(sentence)
            self.cache[sentence] = embedding
            return embedding
