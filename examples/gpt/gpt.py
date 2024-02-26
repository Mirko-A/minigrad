from random import randint
from dataclasses import dataclass

from minitorch.tensor import Tensor
from minitorch.nn.module import Module, Linear, Relu, CrossEntropyLoss, LayerNorm, Embedding, PositionalEncoding, Sequence, MultiHeadAttention
from minitorch.nn.optim import Adam

import time

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

def get_batch(split: str = 'train', batch_size: int = 16, max_context_len: int = 128):
    # data = train_data if split == 'train' else val_data
    data_len = n if split == 'train' else text_len - n
    ix = [randint(0, data_len - max_context_len) for _ in range(batch_size)]
    x = Tensor.concat(0, *[Tensor([encoded_text[i:i+max_context_len]]) for i in ix])
    y = Tensor.concat(0, *[Tensor.concat(0, *[Tensor.one_hot(vocab_size, j).reshape(1, vocab_size)\
        for j in encoded_text[i+1:i+max_context_len+1]]) for i in ix]).reshape([batch_size, max_context_len, vocab_size])
    # print(f'X: {x}')
    # print(f'Y: {y}')  
    
    # y = Tensor.concat(0, *[Tensor([encoded_text[i+1:i+max_context_len+1]]) for i in ix])
    return x, y

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        X, Y = get_batch(split)
        for iter in range(eval_iters):
            print(f'    Eval iteration ({split}): {iter}')
            logits, loss = model(X, Y)
            losses.append(loss.item())
        losses = Tensor(losses)
        out[split] = losses.mean()
    model.train()
    return out

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
    def __init__(self, config: GPTConfig):
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
        pos_enc = self.pos_encoding(T)
        x = tok_emb + pos_enc
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            l = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T, C)
            softmax = logits.softmax()
            l = self.loss(softmax, targets)

        return logits, l
    
    def generate(self, idx: Tensor, max_new_tokens: int, max_context_len: int) -> Tensor:
        for _ in range(max_new_tokens):
            token_cnt = idx.shape[1]
            idx_cond = idx.shrink(1, (token_cnt - max_context_len, 0))
            logits, loss = self(idx_cond)
            timestep_cnt = logits.shape[1] # Take T from (B, T, C)
            logits = logits.shrink(1, (timestep_cnt - 1, 0))
            probs = logits.softmax()
            # TODO: Ivan, 24.2.2024.
            # Create multinomial function
            idx_next = probs.multinomial(num_samples=1)
            idx = Tensor.concat(1, idx, idx_next)
        
        return idx
    
    def params(self) -> list[Tensor]:
        return self.embedding.params() + self.blocks.params() + self.ln_f.params() + self.lm_head.params()
    
    def __call__(self, idx: Tensor, targets: Tensor = None):
        return self.forward(idx, targets)
    
config = GPTConfig(max_context_len=128, vocab_size=vocab_size, n_layer=4, n_head=4, embedding_dim=32)
model = GPT(config)
adam = Adam(model.params(), 0.05)
max_iters = 500
eval_iters = 20
eval_interval = 50

for iter in range(max_iters):
    print(f'Training iteration: {iter}')
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']}, val loss {losses['val']}")
    
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    adam.zero_grad()
    loss.backward()
    adam.step()

context = Tensor.zeros([1])
generated_text = model.generate(context, max_new_tokens=200, max_context_len=64).flatten().data()
generated_text = [int(idx) for idx in generated_text]
print(decode(generated_text))