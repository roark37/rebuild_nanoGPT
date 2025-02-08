import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    dropout: float = 0.1    # no detail is mentioned @GPT2, HF_gpt2 uses 0.1
    bias: bool = True       # True @GPT2, but False is faster and better
    flash: bool = False     # flag of flash attention

@dataclass
class OptConfig:
    weight_decay: float = 0.1    # coefficient of weight decay
    device_type: str = 'cpu'     # need to be specified if cuda is used
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

class LayerNorm(nn.Module):
    """
    self-defined wrapper of LayerNorm, used for the convenience of setting
    the value of argument bias in layernorm all at once.
    """
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim) if bias else None)
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # 作为flag，用于处理residual variance grows
        # dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout # used in flash setting

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = config.flash

        if not self.flash:
            # flash method has causual setting and no need to do this manually
            T = config.block_size
            self.register_buffer('mask', torch.tril(torch.ones(T, T)).view(1, 1, T, T))
        

    def forward(self, x):
        # B:batch size, T:sequence length, C:#channels = n_embd
        B, T, C = x.size()
        
        # nh is "number of heads", hs is "head size", and C () = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) # 第1个参数设置切割出来的每份的size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, 
                                               dropout_p=self.dropout if self.training else 0, 
                                               is_causal=True) 
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))        
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
                
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble heads side by side
        
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # 作为flag，用于处理residual variance grows

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  
        self.attn = CausalSelfAttention(config)  
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config, opt_config):
        super().__init__()
        self.config = config
        self.opt_config = opt_config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # gpt2 has 12 self-attention layers
            ln_f = LayerNorm(config.n_embd, bias=config.bias),  # gpt2 add a layer norm after all attentions and before the last linear(for scores)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # get scores

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights) 

        # report the number of parameters
        print(f'Number of parameters: {self.get_num_params() / 1e6:.2f}M')

    def _init_weights(self, module):
        """
        only two kinds of layers have parameters: linear and embedding
        - linear layers used in projection: need to deal with the residual variance expanding problem
        - other linear layers: std=0.02
        - embedding layers: std=0.02
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # pytroch中bias初始值的default不是0，是一种均匀分布
        elif isinstance(module, nn.Embedding):
            # token embeddings的初始值是正态分布，std=0.02 @gpt2；
            # position embedding正态分布，std=0.01 @gpt2（这里简化都是0.02）
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        return the number of parameters in the model.
        If 'non_embedding=True', the position embeddings get substracted.
        The token embedding is remained because weight sharing.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb               # add drop
        x = self.transformer.drop(x)

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    # customize an optimizer for adding weight decay
    def configure_optimizer(self):
        weight_decay=self.opt_config.weight_decay
        device_type = self.opt_config.device_type
        betas = (self.opt_config.beta1, self.opt_config.beta2)
        eps = self.opt_config.eps

        # construct a Dict for all require grad parameters in the model
        param_dict = {pname: p for pname, p in self.named_parameters() if p.requires_grad}

        # different decay schedule for different params
        # 1. all 2D parameters will be weighted decayed. including weights in matmuls + embeddings
        # 2. others not. including bias and params in payernorms
        decay_params = [p for pname, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pname, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # some old version of pytorch do not support fused argument, so check it before use it
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'# fused would make code fast
        print(f'using fused AdamW: {use_fused}')
        # in the old version of pytorch, the fused argument will be treated as a no-op
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, eps=eps, fused=use_fused)
        # 'lr' arg in AdamW is not set here, it will be set in the Train module
        return optimizer
