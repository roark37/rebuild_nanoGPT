import tiktoken
import torch
import numpy as np
import os

# the base class for dataloaders
class DataLoader():
    def __init__(self, B, T, proc_rank, nproc, mode):
        self.batch_size = B
        self.seq_len = T
        self.proc_rank=proc_rank
        self.nproc=nproc
        assert mode in {'train', 'val'}
        self.mode = mode

    def reset_starting_point(self, shard=None, posi=None):
        """
        this method can be overrided in the child class.
        """
        pass

    def next_batch(self):
        """
        need to be implemented in the child class
        """
        raise NotImplementedError("Must implement next_batch!")
        
        
# dataloader for OpenWebText dataset
class OWDataLoader(DataLoader):
    def __init__(self, B, T, proc_rank, nproc, mode, is_master=False):
        super().__init__(B, T, proc_rank, nproc, mode)

        self.data_root = '../data/openwebtext/'
        if mode == 'train':
            data_path = os.path.join(self.data_root, 'train.bin')
        elif mode == 'val':
            data_path = os.path.join(self.data_root, 'val.bin')
        else:
            raise ValueError("Wrong type of mode! choose from 'train' and 'val'!")

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
     
    def next_batch(self):
        # random sample index
        idx = torch.randint(len(self.data) - self.seq_len, (self.batch_size, ))
        x = [torch.from_numpy((self.data[i: i+self.seq_len]).astype(np.int64)) for i in idx]
        x = torch.stack(x, dim=0)
        y = [torch.from_numpy((self.data[i+1: i+1+self.seq_len]).astype(np.int64)) for i in idx]
        y = torch.stack(y, dim=0)        

        return x, y
        

# load one shard of data and convert it to pytorch tensor
def load_tokens(file_name):
    np_ar = np.load(file_name)
    np_ar = np_ar.astype(np.int32)
    pt_tensor = torch.tensor(np_ar, dtype=torch.long)
    return pt_tensor

## dataloader for FineWeb-EDU dataset
class FWDataLoader(DataLoader):
    def __init__(self, B, T, proc_rank, nproc, mode, is_master=False):
        super().__init__(B, T, proc_rank, nproc, mode)

        # get the shard filenames
        self.data_root = '../data/edu_fineweb10B'
        shards = os.listdir(self.data_root) # 返回指定的文件夹包含的文件名\文件夹名列表
        shards = [s for s in shards if self.mode in s]
        shards = sorted(shards)
        shards = [os.path.join(self.data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f'no shards found for {self.mode}'
        if is_master:
            print(f'found {len(self.shards)} shards for {self.mode}')

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
class SPDataLoader():
    def __init__(self, B, T, proc_rank, nproc, mode='train'):
        super().__init__(B, T, proc_rank, nproc, mode)

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
        
    