import torch
from torch.nn import functional as F

def generate_samples(model, prmpt, n_sample_seq, max_length, 
                     rank=0, device='cpu', device_type='cpu'):
    """
    directly sample from model. No KV cache optimization has been implemented.

    inputs:
    @prmpt: numpy array of encoded tokens

    @ctx_tokens: naive ouput token sequence of max_length.
    """
    
    tokens = torch.tensor(prmpt, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(n_sample_seq, 1)
    ctx_tokens = tokens.to(device)
    # set random seed for torch.multinomial()
    sample_randgen = torch.Generator(device=device)
    sample_randgen.manual_seed(42 + rank)

    while ctx_tokens.size(1) < max_length:
        # forward to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(ctx_tokens)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50(HuggingFace pipeline default)
              # torch.topk: Returns the k largest elements of input along the given dim
              # A namedtuple of (values, indices) is returned with the values and indices
              # of the largest k elements of each row of the input in the given dim.
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
              # sample one token use multinomial distribution, return the indices
            tar_idx = torch.multinomial(topk_probs, 1, generator=sample_randgen)# (B,1)
              # gather the corresponding indicies
            tar_tok = torch.gather(topk_indices, -1, tar_idx) # (B, 1)
            # append to the context
            ctx_tokens = torch.cat((ctx_tokens, tar_tok), dim=1)  

    return ctx_tokens 
