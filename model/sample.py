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
tokens = enc.encode("In 1842, the Bishop of Vincennes, Célestine Guynemer de la Hailandière, offered land to Father Edward Sorin of the Congregation of the Holy Cross, on the condition that he build a college in two years. Fr. Sorin arrived on the site with eight Holy Cross brothers from France and Ireland on November 26, 1842, and began the school using Father Stephen Badin's old log chapel. He soon erected additional buildings, including Old College, the first church, and the first main building. They immediately acquired two students and set about building additions to the campus. In what year was Father Edward Sorin given two years to create a college?")
tokens = torch.tensor(tokens,dtype=torch.long)
x = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = x.to('cuda')
x = model.generate(x,max_new_tokens=max_length,temperature=0.95,top_k=100)


for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)