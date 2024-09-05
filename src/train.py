import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 100
n_embd = 128
n_head = 4
n_layer = 6
dropout = 0.4
print(device)
print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Version:", torch.version.cuda)
# ------------

with open('Dataset/Haruki Murakami/HM.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
print(f"Vocab_size: {vocab_size}")


# Train and test splits
data = enc.encode(text)
data = torch.tensor(data)
data = data.to(device)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
  out ={}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  #single head of attention
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) # (B,T,head_size)
    q = self.query(x) #(B,T,head_size)

    wei = q @ k.transpose(-1,-2) * C ** -0.5 #(B,T,head_size) @ (B,head_size,T) = (B,T,T)
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #(B,T,T)
    wei = F.softmax(wei,dim=-1)
    wei = self.dropout(wei)

    v = self.value(x) #(B,T,head_size)
    out = wei @ v
    return out

class MultiHeadedAttention(nn.Module):
  def __init__(self,num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) #(B,T,C)
    out = self.dropout(self.proj(out))
    return out


class FeedForward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd,4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd)
    )

  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadedAttention(n_head,head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self,x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


class LanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)

  def forward(self,idx, targets = None):
    B,T = idx.shape

    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
    x = tok_emb + pos_emb #(B,T,C)
    x = self.blocks(x) #(B,T,C)
    x = self.ln_f(x) #(B,T,C)
    logits = self.lm_head(x) #(B,T,vocab_size)

    if targets is not None:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits,targets)
    else:
      loss = None
    return logits, loss


  def generate(self,idx, max_new_tokens):
    for _ in range(max_new_tokens):
      #crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)

      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim=-1)
      idx_next = torch.multinomial(probs,num_samples=1)
      idx = torch.cat((idx,idx_next), dim=1)
    return idx


model = LanguageModel()
# state_dict = torch.load('Pre-trained/model_params_2.pth')
# model.load_state_dict(state_dict)
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

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
model_save_path = 'Pre-trained/model_params_HM.pth'
torch.save(model.state_dict(),model_save_path)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(enc.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))