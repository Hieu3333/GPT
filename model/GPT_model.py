import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self,ndim,bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self,input):
        return F.layer_norm(input,self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3* config.n_embd,bias = config.bias)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        #regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        #flash_attention but only in torch >= 2.0
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            print("Flash attention not supported")
            #Use causal mask for slow attention
            self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))

        def forward(self,x):
            B,T,C = x.shape 
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q,k,v = self.c_attn(x).split(self.n_embd, dim = 2)
            k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # (B, n_head, T, head_size)
            q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # (B, n_head, T, head_size)
            v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # (B, n_head, T, head_size)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                out = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2,-1)) * (1.0 /math.sqrt(k.shape[-1]))
                att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
                att = F.softmax(att,dim=-1)
                att = self.attn_dropout(att)
                out = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            out = out.transpose(1,2).contiguous().view(B,T,C) # re-assemble all head outputs side by side

            #output projection
            out = self.resid_dropout(self.c_proj(out))
            return out

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias = config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd,bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias = False)

        #weight tying
        self.transformer.wte.weight = self.lm_head.weight

        #init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0, std = 0.02/math.sqrt(2*config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self,non_embedding = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if (isinstance(module,nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0, std = 0.02)

    def forward(self,idx, targets = None):
        device = idx.device
        B,T = idx.shape
        pos = torch.arange(0,T,dtype = torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(idx)
        x = self.transformer.drop(tok_emb+pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]), targets.view(-1),ignore_index=-1 )

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:,[-1],:]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss
    