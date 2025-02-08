import tiktoken
import torch
import numpy as np
import os

# load one shard of data and convert it to pytorch tensor
def load_tokens(file_name):
    np_ar = np.load(file_name)
    np_ar = np_ar.astype(np.int32)
    pt_tensor = torch.tensor(np_ar, dtype=torch.long)
    return pt_tensor

## dataloader for FineWeb-EDU dataset
class DataLoader():
    def __init__(self, B, T, proc_rank, nproc, mode, is_master=False):
        self.batch_size = B
        self.seq_len = T
        self.proc_rank=proc_rank
        self.nproc=nproc
        assert mode in {'train', 'val'}

        # get the shard filenames
        data_root = '../data/edu_fineweb10B'
        shards = os.listdir(data_root) # 返回指定的文件夹包含的文件名\文件夹名列表
        shards = [s for s in shards if mode in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f'no shards found for {mode}'
        if is_master:
            print(f'found {len(self.shards)} shards for {mode}')

        self.reset_starting_point()

    def reset_starting_point(self, shard=0, posi=0):
        self.curr_shard_idx = shard
        # load the current shard
        self.tokens = load_tokens(self.shards[self.curr_shard_idx])
        self.curr_posi = self.batch_size * self.seq_len * self.proc_rank + posi

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        buf = self.tokens[self.curr_posi : self.curr_posi + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1: ]).view(B, T)
        self.curr_posi += B * T * self.nproc
        # if the next patch would be out of bounds, advance to the next shard
        if self.curr_posi + (B * T * self.nproc + 1) > len(self.tokens):
            self.curr_shard_idx = (self.curr_shard_idx + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.curr_shard_idx])
            self.curr_posi = B * T * self.proc_rank

        return x, y


## dataloader for shakespeare txt in 'data/input.txt'
class DataLoaderLite():
    def __init__(self, B, T, proc_rank, nproc):
        self.batch_size = B
        self.seq_len = T
        self.proc_rank=proc_rank
        self.nproc=nproc

        with open('../data/input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T*nproc)} batches')

        self.current_position = B * T * self.proc_rank

    def next_batch(self):
        B, T, N = self.batch_size, self.seq_len, self.nproc
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        
        # advance the position in the tensor
        self.current_position += B * T * N
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * N + 1) > len(self.tokens):
            self.current_position = B * T * self.proc_rank
        return x, y
        
    