
import math
import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataloader import DataLoaderLite, DataLoader

class TrainerConfig:
    # lr schedule params
    max_lr = 6e-4
    min_lr = 0.1 * max_lr
    max_steps = 5000      # 19073
    warmup_steps = 200     # 715

    # file path for saving and loading
    log_dir = '../log'     # directory for log file and checkpoint files
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log.txt')
    with open(log_file, 'w') as f:
        pass
    
    def __init__(self, world_size=1, rank=0, local_rank=0, master=True, 
                 device='cpu', device_type='cpu', **kwargs):
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.device = device
        self.master = master
        self.device_type = device_type
        for k, v in kwargs.items():
            setattr(self, k, v)

        assert self.total_bsize % (self.B * self.T * self.world_size) == 0
        self.grad_accum_steps = self.total_bsize // (self.B * self.T * self.world_size)


class Trainer():
    def __init__(self, model, optimizer, config, ckpt_path=None):
        self.ckpt_path = ckpt_path
        self.optimizer = optimizer
        self.config = config
        self.train_loader = DataLoader(config.B, config.T, 
                                       proc_rank=config.rank, 
                                       nproc=config.world_size, 
                                       mode='train',
                                       is_master=config.master
                                       )
        self.val_loader = DataLoader(config.B, config.T, 
                                     proc_rank=config.rank, 
                                     nproc=config.world_size, 
                                     mode='val',
                                     is_master=config.master
                                     )
        
        # DDP setting
        self.ddp = int(os.environ.get('RANK', -1)) > -1
        self.model = DDP(model) if self.ddp else model
        self.raw_model = self.model.module if self.ddp else model

        # setting the starting point of training
        self.steps_run = 0
        if ckpt_path is not None:
            assert os.path.exists(ckpt_path), 'No checkpoint file to be loaded!'
            self._load_checkpoint(ckpt_path)

        
    def _load_checkpoint(self, ckpt_path):
        dist.barrier()     # make sure there are no load before saving
        from_master_to_local = { f'cuda:0': f'cuda:{self.config.local_rank}'} 
        snapshot = torch.load(ckpt_path, map_location=from_master_to_local)
        self.raw_model.load_state_dict(snapshot['model'])
        self.optimizer.load_state_dict(snapshot['optimizer'])
        self.steps_run = snapshot['step']
        shard_idx, inshard_posi = snapshot['data_position']
        self.train_loader.reset_starting_point(shard=shard_idx, posi=inshard_posi)
        if self.ddp:
            dist.barrier() # wait for all local ranks to finish the loading
        print(f'rank {self.config.rank} resuming training from checkpoint at {ckpt_path},\
               from step {self.steps_run}.')

    def _save_checkpoint(self, checkpoint_path, step, 
                        shard_idx, inshard_posi, val_loss_accum):
        checkpoint = {
                        'model': self.raw_model.state_dict(), 
                        'optimizer' : self.optimizer.state_dict(),
                        'config': self.raw_model.config,
                        'step': step,
                        'data_position': (shard_idx, inshard_posi), 
                        'val_loss': val_loss_accum.item(),
                     }
        torch.save(checkpoint, checkpoint_path)
    
    def _get_lr(self, curr_step):
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps
        max_lr = self.config.max_lr
        min_lr = self.config.min_lr

        # 1) linear warmup in the first warmup_steps
        if curr_step < warmup_steps:
            return max_lr * (curr_step + 1) / warmup_steps
        # 2) when the current step reachs the lr_decay_iters, lr stay at min_lr
        if curr_step > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min_lr
        decay_ratio = (curr_step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def _step(self, step_idx, mode, accum_steps=1):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # for log and visualization
        loss_accum = 0.0
        norm = None

        # make a minibatch of training data
        for micro_step in range(accum_steps):

            # 1) get one mini-batch of data
            if mode == 'train':
                x, y = self.train_loader.next_batch()
            else:
                x, y = self.val_loader.next_batch()
            x, y = x.to(self.config.device), y.to(self.config.device)

            with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            
            # 2) # make up for the accumulation ops                                 
            loss /= accum_steps
            
            # 3) visualization info.
            loss_accum += loss.detach()  # for visualization
            
            # 4) get gradient and control ddp synchronization
            if mode == 'train':
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == accum_steps - 1)
                loss.backward()

        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # synch aross gpus
        if mode =='train':
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # should be the same to use 'self.raw_model.parameters()' as the argument

            # gradient update
            lr = self._get_lr(step_idx)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()

        # wait for all kernels in all streams on a cuda device to complete
        torch.cuda.synchronize()
        return loss_accum, norm

    def train(self):
        """
        run optimization to train the model
        """
        model = self.model
        raw_model = self.raw_model
        optimizer = self.optimizer
        config = self.config

        for step_i in range(self.steps_run, config.max_steps):
            t0 = time.time()
            last_step = (step_i == config.max_steps -1)

            # evaluating the validation loss and checkpointing
            if step_i % 50 ==0 or last_step:
                raw_model.eval()
                mode = 'eval'
                self.val_loader.reset_starting_point()
                with torch.no_grad():
                    loss_accum, _ = self._step(step_idx=step_i, mode=mode, accum_steps=20)

                if config.master:
                    print(f'val loss:{loss_accum.item():.4f}')
                    with open(config.log_file, 'a') as f:
                        # log validation loss
                        f.write(f'{step_i} val {loss_accum.item():.4f}\n')
                
                    # write model checkpoints
                    ckpt_path = os.path.join(config.log_dir, f'model_{step_i:05d}.pt')
                    shard_idx = self.train_loader.curr_shard_idx
                    inshard_posi = self.train_loader.curr_posi
                    self._save_checkpoint(ckpt_path, step_i, shard_idx, inshard_posi, loss_accum)    

            # one step of optimization
            raw_model.train()
            mode = 'train'
            optimizer.zero_grad()
            loss_accum, norm = self._step(step_idx=step_i, mode=mode, accum_steps=config.grad_accum_steps)

            # timing
            t1 = time.time() # return value is measured by seconds
            dt = (t1 - t0) * 1000 # measured using ms
            tokens_processed = config.total_bsize
            tok_per_sec = tokens_processed / (t1 - t0) / 1000
            if config.master:
                print(f'step:{step_i}, loss:{loss_accum.item():.2f}| norm:{norm:.2f}| dt:{dt:.0f}ms| tok/sec:{tok_per_sec:.1f}k')
