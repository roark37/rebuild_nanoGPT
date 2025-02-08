"""
Downloads and tokenizes the FineWeb-Edu dataset
the original FineWeb-Edu dataset has 1.3T tokens, 
here we use the smallest sample version 'sample-10BT'
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Run:
$ python fineweb-edu.py
will save shards to the local directory '../data/edu_fineweb10B'
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
                                  # huggingface datasets lib
from tqdm import tqdm             # pip install tqdm



fineweb_edu_version = 'sample-10BT' # the smallest sample version of fineweb-edu
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# make directory
local_dir = '../data/edu_fineweb10B'
os.makedirs(local_dir, exist_ok=True)

# download dataset from huggingface library
# when specified the split arg, the load_dataset returns a datasets.Dataset objec
fw_data = load_dataset('HuggingFaceFW/fineweb-edu', name=fineweb_edu_version, split='train')
# the structure of fw_data: each row of 'text' is a piece of document
# Dataset({
#     features: ['text', 'id', 'dump', 'url', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'],
#     num_rows: 9672101
# })

# tokenizer
enc = tiktoken.get_encoding('gpt2') # fineweb also uses the GPT2 tokenizer
end_of_txt = enc._special_tokens['<|endoftext|>'] # add one special token '<eot>'

def tokenize(doc):
    """
    tokenize a single doc, returns a numpy array of uint16 tokens
    """
    # use '<eot>' to delimits all documents
    tokens = [end_of_txt]

    # put a '<eot>' at the front of every documents
    # unlike enc.encode, enc.encode_ordinary would ignore special tokens
    # it's used to encode the original text, no special token is here
    tokens.extend(enc.encode_ordinary(doc['text']))

    # convert dtype of tokens
    tokens_np = np.array(tokens)

    # check if the value of tokens are in the range of np.uint16
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'token dict too large for uint16'
    tokens_np_uint16 = tokens_np.astype(np.uint16)

    return tokens_np_uint16

def wirte_datafile(filename, tokens_np):
    """
    save the numpy array to a local file
    """
    np.save(filename, tokens_np)

# tokenize all docs and write outputs in shards
# each shard has shard_size tokens except the last shard has remainder
nprocs = max(1, os.cpu_count()//2)   # cpu_count() -> the cpu's number of threads
                                     # specify half of threads as workers
print(f'{nprocs} procs are(is) working')

with mp.Pool(nprocs) as pool:
    shard_idx = 0

    # pre-allocate a buffer of np array to hold current shard
    curr_shard_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    # iterate and tokenize through each doc in the dataset
    for tokens in pool.imap(tokenize, fw_data, chunksize=16):
        # imap(): multi-processing and lazy version of map()
        # https://superfastpython.com/multiprocessing-pool-imap/
        if token_count + len(tokens) < shard_size:
            # append tokens to current shard
            curr_shard_np[token_count: token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'shard{shard_idx}')
            progress_bar.update(len(tokens))
        else:
            # split the document as two parts, the 1st fit to the current shard
            # the remainder goes to the next shard
            first_len = shard_size - token_count
            curr_shard_np[token_count: token_count+first_len] = tokens[:first_len]

            # write the current shard to disk
            split = 'val' if shard_idx == 0 else 'train' # 1st -> validation set
            file_name = os.path.join(local_dir, f'edufineweb_{split}_{shard_idx:06d}')
            wirte_datafile(file_name, curr_shard_np)

            # start a new shard(update index) and reset the progress bar
            shard_idx += 1
            progress_bar = None

            # populate the leftovers of the current doc to the new shard
            second_len = len(tokens)-first_len
            curr_shard_np[:second_len] = tokens[first_len:]
            token_count = second_len
    
    # write the last shard if there are not enough remaining tokens for a full shard
    if token_count != 0:
        split = 'val' if shard_idx == 0 else 'train'
        file_name = os.path.join(local_dir, f'edufineweb_{split}_{shard_idx:06d}')
        wirte_datafile(file_name, curr_shard_np[:token_count])
            









