import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# hyperparameters
batch_size = 64 
max_iters = 20000
eval_interval = 500
learning_rate = 1e-5
device = 'cpu' #'cuda'
eval_iters = 200
n_embd = 768
n_head = 1
n_layer = 1
# ------------

torch.manual_seed(1337)

data = np.load("data/in.npy")
labels = np.load("data/out.npy")
ds = data.shape
data = data.reshape((ds[0]*ds[1], ds[2], ds[3]))

# Train and test splits
data = torch.tensor(data, dtype=torch.float)
labels = torch.tensor(labels, dtype=torch.float)
ldata = 50000 #len(data)
print(f"Length: {ldata}")
train_data = data[:int(ldata)]
val_data = data[int(ldata):ldata+1000]
train_labels = labels[:int(ldata)]
val_labels = labels[int(ldata):ldata+1000]

print(f"Labels: {labels.shape}")
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    x,y = None, None
    if split == 'train':
        ri = torch.randperm(len(train_data))[:batch_size]
        x = train_data[ri, :]
        y = train_labels[ri, :]
    elif split == 'test':
         ri = torch.randperm(int(len(data)))[:]
         x = data[ri, :]
         y = labels[ri, :]
    elif split == 'val':
         ri = torch.randperm(int(len(val_data)))[:batch_size]
         x = val_data[ri, :]
         y = val_labels[ri, :]

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # Get KQ weights
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = 64 #n_embd // n_head
        self.sa = Head(head_size) #MultiHeadAttention(n_head, head_size)

    def forward(self, x):
        return self.sa(x)

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        #self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        #self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        #self.lossf=  nn.NLLLoss()
        #self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        #print(idx.shape)
        B, T, C = idx.shape

        # idx and targets are both (B,T) tensor of integers
        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C)
        logits = self.blocks(idx) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        #logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            #print(logits.shape, targets.shape)
            B, T, C = logits.shape
            logits = logits.view(B, T*C)
            targets = targets.view(B, T*C)
            loss = F.mse_loss(logits, targets)
        return logits, loss


model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()), 'parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
