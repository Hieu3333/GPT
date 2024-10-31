import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from Config.config import GPTConfig
from RMS_Norm import RMSNorm
import inspect

class SwiGLU(nn.Module):
    def __init__(self,fan_in,fan_out):
        super(SwiGLU,self).__init__()
        #Single linear layer with double the output size for splitting
        self.linear = nn.Linear(fan_in,fan_out*2)
        self.activation = nn.SiLU()

    def forward(self,x):
        x= self.linear(x)
        x_gated,x_activated = x.chunk(2,dim=-1)
        return self.activation(x_activated) * x_gated


def lambda_init_fn(depth):
    return 0.8 - 0.6 * torch.exp(-0.3*(depth-1))


class DiffAttn(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd / config.n_head
        self.d = self.head_size / 2 
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.query = nn.Linear(self.n_embd, self.head_size, bias=False) #(C,2d)
        self.key = nn.Linear(self.n_embd, self.head_size, bias = False) #(C,2d)
        self.value = nn.Linear(self.n_embd, self.head_size, bias = False) #(C,2d)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.d, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.d, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.d, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.d, dtype=torch.float32).normal_(mean=0,std=0.1))

        
        self.attn_dropout = nn.Dropout(self.dropout)

        self.register_buffer('bias',torch.tril(torch.ones(config.block_size, config.block_size))) #(T,T)

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x) #(B,T,head_size==2d)
        k = self.key(x) #(B,T,head_size==2d)
        v = self.value(x) #(B,T,head_size==2d)

        q1,q2 = q.split(self.d,dim=-1) #(B,T,d)
        k1,k2 = k.split(self.d,dim=-1) #(B,T,d)
        #Value is the same

        #Differential attention
        att1 = (q1 @ k1.transpose(-2,-1)) * 1.0/(math.sqrt(self.d)) #(B,T,T)
        att1 = att1.masked_fill(self.bias[:,:T,:T]==0, float('-inf'))
        att1 = F.softmax(att1, dim=-1)

        att2 = (q2 @ k2.transpose(-2,-1)) * 1.0/(math.sqrt(self.d)) #(B,T,T)
        att2 = att2.masked_fill(self.bias[:,:T,:T]==0, float('-inf'))
        att2 = F.softmax(att2, dim=-1)

        lambda1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda1 - lambda2 + self.lambda_init

        att_full = att1 - lambda_full*att2
        att_full = self.attn_dropout(att_full)
        att_full = att_full @ v #(B,T,T) @ (B,T,2d) -> (B,T,2d)

        return att_full

class MultiHeadDiffAttn(nn.Module):
    def __init__(self,config,depth):
        super().__init__()
        self.heads = nn.ModuleList([DiffAttn(config,depth) for _ in range(config.n_head)]) #(B,T,C)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.resid_dropout(self.out_proj(out))
        return out
        
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.swiglu = SwiGLU(config.n_embd, 2*config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,config,depth):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd,bias=config.bias)
        self.attn = MultiHeadDiffAttn(config,depth)
        self.ln_2 = RMSNorm(config.n_embd, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) # Keep the norm layer before the attention layer, unlike in the paper they use norm after the attention?
        x = x + self.mlp(self.ln_2(x))
        return x

class DifferentialGPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.diff_transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config,depth) for depth in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.diff_transformer.wte.weight = self.lm_head.weight

        #init weight
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0, std = 0.02/math.sqrt(2*config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, idx, target=None):
        device = idx.device
        B,T = idx.shape
        pos = torch.arange(0,T,dtype=torch.long,device=device)

        tok_emb = self.diff_transformer.wte(idx)
        pos_emb = self.diff_transformer.wpe(idx)

        x = self.diff_transformer.drop(tok_emb+pos_emb)

        for block in self.diff_transformer.h:
            x = block(x)
        x = self.diff_transformer.ln_f(x)

        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.shape[-1]),target.view(-1))
        else:
            logits = self.lm_head(x)
            loss = None

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
    
    def configure_optimizer(self,weight_decay,learning_rate, betas, device_type):
        param_dict = {pn: p for pn,p in self.named_parameters()}
         # filter out those that do not require grad
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad }
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim()<2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:,-self.config.block_size:]
            logits, _ = self(idx_cond)
            #get the logits and scale by desired temperature
            logits = logits[:,-1,:] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v,_ = torch.topk(logits, min(top_k,logits.shape[-1]))
                logits[logits < v[:,[-1]]] = -float('Inf')
            probs = F.softmax(logits,dim=-1)
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1)
            import tiktoken
            enc = tiktoken.get_encoding('gpt2')
            if idx_next == enc.eot_token:
                break
            idx = torch.cat((idx,idx_next), dim=1)
        return idx
