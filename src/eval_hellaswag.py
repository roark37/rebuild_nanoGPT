"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

- Example HellaSwag json item:
{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

- Key items in the example:
'ind': dataset ID
'activity_label': The ActivityNet or WikiHow label for this example
'context': There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
'endings': a list of 4 endings. The correct index is given by label (0,1,2, or 3)
'split': train, val, or test.
'split_type': indomain if the activity label is seen during training, else zeroshot
'source_id': Which video or WikiHow article this example came from
'label'(examples in test set do not have this item): The correct index with a value of 0, 1, 2, or 3.

- The way to choose the answer from the LM:
  - pick the endings with the lowest average loss(per token loss)

- The validation set of HellaSwag has a total of 10,042 examples.

- Benchmark:
  - gpt2 (124M)
    - eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
    - this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

  - gpt2-xl (1558M)
    - eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
    - this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)
"""
import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
import torch.distributed as dist
from transformers import GPT2LMHeadModel

## download the dataset and wrap the examples into iterables
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

DATA_CACHE_DIR = "../data/hellaswag"

# helper function used in download()
def download_from_url(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download(split):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_from_url(data_url, data_filename)

# wrap all examples from a split into an iterable
def iterate_examples(split):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

## render the format of the example for easy processing with GPT model 
enc = tiktoken.get_encoding('gpt2')

def render_example(example):
    """
     Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood) 

    outputs:
    @data: a dict of 'label', 'ctx_tokens' and 'ending_tokens'

    @tokens: torch tensor with shape (4, T)
             there are four endings in each example;
             tokenize (ctx + ending_i) get four tokens sequences
             T is the max of the length of the four tokens sequences
             Those shorter sequence is padded with zeros in the end

    @mask: torch tensor with shape (4, T), same as @tokens
           positions corresponding to the tokens of endings are assigned a value of 1,
           while all others are assigned 0.
           
    """
    ctx = example['ctx']          # str
    label = example['label']      # int
    endings = example['endings']  # list of str

    data = {
        'label' : label, 
        'ctx_tokens': None,
        'ending_tokens': [] 
    }

    # tokenize ctx and endings
    ctx_tokens = enc.encode(ctx)
    data['ctx_tokens'] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(' ' + end) # add ' ' because GPT2 tokenizer
                                           # encode 'x' differently from ' x'
                                           # ' x' is the right format in between a sentence
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data['ending_tokens'].append(end_tokens)

    # pad token and mask rows to the same length
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def get_most_likely_row(logits, tokens, mask):
    # cut the output of the last token
    shift_logits = (logits[...,:-1,:]).contiguous() 
    # the output sequence do not have the first token
    shift_tokens = (tokens[..., 1:]).contiguous()  # shape: (4, (T-1))
    shift_mask = (mask[:, 1:]).contiguous()      # shape: (4, (T-1))
            
    # flat for cross entropy function
    # shape of logits: (4*(T-1), vocab_size), T is the row length in tokens
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    # shape of tokens: (4*(T-1),)
    flat_shift_tokens = shift_tokens.view(-1)
    # set the reduciton argument, otherwise loss will be 'mean loss'
    shift_loss = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')

    shift_loss = shift_loss.view(tokens.size(0), -1) # shape: (4, (T-1))
    masked_shift_loss = shift_loss * shift_mask
    avg_loss = masked_shift_loss.sum(dim=1) / shift_mask.sum(dim=1)
    pred_label = avg_loss.argmin().item()

    return pred_label

## ----- compute the predict accuracy for a pretrained model----- ##
def predict(model, 
            device='cpu', device_type='cpu',
            ddp=False, rank=0, world_size=1,
            hf_pretrain=False, 
            eval_book_file=None):
    """
    return the predict labels from the model and the correct labels

    inputs:
    @hf_pretrain: True when using HuggingFace GPT2 model.
                  Its output api is different from our model
    @eval_book_file (None or str): if it's not None, then the predict
                                   label will be logged in the file
    """

    correct_num = 0
    total_num = 0
    model.eval()

    for i, example in enumerate(iterate_examples('val')):
        if i % world_size != rank:
            continue
        # render the example
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if hf_pretrain:  # load pretrained model from huggingface
                    logits = model(tokens).logits  # api of GPT2LMHeadModel.from_pretrained('gpt2')
                else:
                    logits, _ = model(tokens)

            pred_label = get_most_likely_row(logits, tokens, mask)
            if eval_book_file is not None:
                with open(eval_book_file, 'a') as f:
                    f.write(f'{i}, {label}, {pred_label}\n')

            if pred_label == label:
                correct_num += 1
            total_num += 1

    if device_type == 'cuda':
        torch.cuda.synchronize()

    if ddp:
        total_num = torch.tensor(total_num, dtype=torch.long, device=device)
        dist.all_reduce(total_num, op=dist.ReduceOp.SUM)
        total_num = total_num.item()

        correct_num = torch.tensor(correct_num, dtype=torch.long, device=device)
        dist.all_reduce(correct_num, op=dist.ReduceOp.SUM)
        correct_num = correct_num.item()
            
    acc = correct_num / total_num

    return acc


if __name__ == '__main__':
    print(f'evaluate hellswag using HuggingFace GPT2 model: ')
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = device_type = 'cuda'
    acc = predict(hf_model, device, device_type, hf_pretrain=True)
    print(f'acc: {acc:.2f}')