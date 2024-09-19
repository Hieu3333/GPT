from GPT_model import GPT, GPTConfig
import torch
import tiktoken
from torch.nn import functional as F
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class DataLoader():
    def __init__(self,data,config):
        self.data = data
        self.position = 0
        self.B = config.batch_size
        self.T = config.block_size

    def get_batch(self):
        if self.position >= len(self.data):
            self.position = 0
        s = torch.tensor(self.data[self.position:self.position+self.B*self.T+1])
        
        x = s[:self.B*self.T]
        y = s[1:]
        x, y = x.view(self.B,self.T), y.view(self.B,self.T)
        self.position += (self.B*self.T+1)
        return x,y

dataset_path = 'Dataset/Haruki Murakami/HM.txt'
with open(dataset_path,'r',encoding='utf-8') as file:
    dataset = file.read()
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(dataset)
dataloader = DataLoader(tokens,config=GPTConfig)

torch.set_float32_matmul_precision('high')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Train on:',device)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model = model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
max_iters = 10
for i in range(max_iters):
    t0 = time.time()
    x,y = dataloader.get_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0) * 1000 #milliseconds
    tokens_per_sec = (dataloader.B * dataloader.T)/(t1-t0)
    if i%1 == 0:
        print(f"Iter {i}: loss :{loss.item()} dt:{dt:.2f}s tok/sec:{tokens_per_sec:.2f}")
    





