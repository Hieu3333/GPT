import torch
from GPT_model import GPT, GPTConfig
from torch.nn import functional as F
import tiktoken
import warnings
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")


num_return_sequences = 1
max_length = 100

model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig())
model.eval()
model = model.to('cuda')

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("I want to tell you a story of a rabbit and a turtle: Once upon a time")
tokens = torch.tensor(tokens,dtype=torch.long)
x = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = x.to('cuda')
x = model.generate(x,max_new_tokens=max_length,temperature=0.95,top_k=100)


for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)