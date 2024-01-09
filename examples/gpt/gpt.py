from random import randint

from minitorch.tensor import Tensor

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
train_data = Tensor([encoded_text[:n]])
val_data = Tensor([encoded_text[n:]])

def get_batch(split: str = 'train', batch_size: int = 32, max_context_len: int = 256):
    data = train_data if split == 'train' else val_data
    data_len = n if split == 'train' else text_len - n
    ix = [randint(0, data_len - max_context_len) for _ in range(batch_size)]
    # TODO: Mirko, 9. 1. 2024.
    # x and y will be 1D Tensors of length 8192 after concatenation
    # if the desired shape is 32x256, please add an additional pair
    # of braces in the Tensor constructor calls. Just a reminder.
    x = Tensor.concat(0, *[Tensor(encoded_text[i:i+max_context_len]) for i in ix])
    y = Tensor.concat(0, *[Tensor(encoded_text[i+1:i+max_context_len+1]) for i in ix])
    return x, y

x, y = get_batch()
print(x)