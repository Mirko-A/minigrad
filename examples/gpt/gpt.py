from random import randint
from dataclasses import dataclass

from minitorch.tensor import Tensor
from minitorch.nn.module import Module, Linear, Relu, CrossEntropyLoss, LayerNorm, Embedding, PositionalEncoding, Sequence, MultiHeadAttention

with open('./examples/gpt/the_sopranos_pilot.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# print(f'Char list:{chars}\nVocab size:{vocab_size}\nEncoded:{encode("Kids! Come here!")}\nDecoded:{decode(encode("Kids! Come here!"))}')

encoded_text = encode(text)
text_len = len(encoded_text)
n = int(0.9*text_len)
# train_data = Tensor([encoded_text[:n]])
# val_data = Tensor([encoded_text[n:]])

def get_batch(split: str = 'train', batch_size: int = 32, max_context_len: int = 256):
    # data = train_data if split == 'train' else val_data
    data_len = n if split == 'train' else text_len - n
    ix = [randint(0, data_len - max_context_len) for _ in range(batch_size)]
    x = Tensor.concat(0, *[Tensor([encoded_text[i:i+max_context_len]]) for i in ix])
    y = Tensor.concat(0, *[Tensor.concat(0, *[Tensor.one_hot(vocab_size, j).reshape(1, vocab_size)\
        for j in encoded_text[i+1:i+max_context_len+1]]) for i in ix]).reshape([batch_size, max_context_len, vocab_size])
    # y = Tensor.concat(0, *[Tensor([encoded_text[i+1:i+max_context_len+1]]) for i in ix])
    return x, y

class Block(Module):
    class FeedFoward(Module):
        def __init__(self, embedding_dim: int):
            self.net = Sequence(
                Linear(embedding_dim, 4 * embedding_dim),
                Relu(),
                Linear(4 * embedding_dim, embedding_dim)
            )

        def forward(self, input: Tensor):
            return self.net(input)
        
        def params(self) -> list[Tensor]:
            return self.net.params()
        
        def __call__(self, input: Tensor) -> Tensor:
            return self.forward(input)
    
    def __init__(self, embedding_dim: int, n_head: int, context_len: int):
        head_size = embedding_dim // n_head
        self.sa = MultiHeadAttention(embedding_dim, n_head, head_size, context_len)
        self.ffwd = Block.FeedFoward(embedding_dim)
        self.ln1 = LayerNorm(embedding_dim)
        self.ln2 = LayerNorm(embedding_dim)

    def forward(self, input: Tensor):
        #? NOTE: Ivan, 7.1.2024.
        # In the original Transformer paper (Attention Is All You Need),
        # add & norm is applied after the transformation (Post-LN), 
        # but here we apply LayerNorm before the transformation (Pre-LN).
        # This can make training more stable.
        input = input + self.sa(self.ln1(input))
        input = input + self.ffwd(self.ln2(input))
        return input
    
    def params(self) -> list[Tensor]:
        return self.sa.params() + self.ffwd.params() + self.ln1.params() + self.ln2.params()
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

@dataclass
class GPTConfig:
    max_context_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    embedding_dim: int

class GPT(Module):
    def __init__(self, config):
        self.embedding = Embedding(config.vocab_size, config.embedding_dim)
        self.pos_encoding = PositionalEncoding(config.embedding_dim)
        self.blocks = Sequence(*[Block(config.embedding_dim, n_head=config.n_head, context_len=config.max_context_len) for _ in range(config.n_layer)])
        #? NOTE: Ivan, 14.1.2024.
        # Since we use Pre-LN, final LayerNorm is applied after the decoder stack. 
        # This is deviates from the original paper.
        self.ln_f = LayerNorm(config.embedding_dim)
        self.lm_head = Linear(config.embedding_dim, config.vocab_size)
        self.loss = CrossEntropyLoss()
        
    def forward(self, idx: Tensor, targets: Tensor = None):
        B, T = idx.shape

        tok_emb = self.embedding(idx)
        pos_enc = self.pos_encoding(range(0, T))
        x = tok_emb + pos_enc
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            l = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            softmax = logits.softmax()
            l = self.loss(softmax, targets)

        return logits, l
    
    def generate(self, idx: Tensor, max_new_tokens: int, max_context_len: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_context_len:]
            logits, loss = self(idx_cond)
    
    def params(self) -> list[Tensor]:
        return self.embedding.params() + self.blocks.params() + self.ln_f.params() + self.lm_head.params()
    
    def __call__(self, idx: Tensor, targets: Tensor = None) -> Tensor:
        return self.forward(idx, targets)
    
config = GPTConfig(max_context_len=256, vocab_size=vocab_size, n_layer=6, n_head=6, embedding_dim=64)