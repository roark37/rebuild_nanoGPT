import argparse
import os
import tiktoken
import torch
from torch.distributed import init_process_group, destroy_process_group

from gpt_model import GPTConfig, GPT
from trainer import TrainerConfig, Trainer
from eval_hellaswag import predict
from sample_utils import generate_samples

# when using ddp, use torchrun:
# for pretrain:
# torchrun --standalone --nproc_per_node=1 run.py pretrain \
#          --writing_params_path pretrain.params

# torchrun --standalone --nproc_per_node=1 run.py pretrain \
#          --writing_params_path pretrain.params \
#          --ckpt_path ../log/model_00499.pt

# for generate:
# torchrun --standalone --nproc_per_node=1 run.py generate \
#          --reading_params_path pretrain.params

# for evaluate:
# torchrun --standalone --nproc_per_node=1 run.py evaluate \
#          --reading_params_path pretrain.params

## ------------ CommandLine parsing ------------
argp = argparse.ArgumentParser()

argp.add_argument('function', default='pretrain', 
                  help="Choose pretrain, finetune, evaluate or generate")

argp.add_argument('--train_data_path', default=None)
argp.add_argument('--ft_data_path', default=None)
argp.add_argument('--eval_data_path', default=None)

argp.add_argument('--writing_params_path', default=None, 
                  help="specify a file to save model.state_dict")
argp.add_argument('--reading_params_path', default=None)
argp.add_argument('--ckpt_path', default=None, 
                  help="start training from the specified checkpoint.")

args = argp.parse_args()


## ------------ Device & Environment setting ------------
ddp = int(os.environ.get('RANK', -1)) > -1
print(f'ddp is activated: {ddp}')
if ddp:
    assert torch.cuda.is_available(), 'no gpu device!'
    init_process_group(backend='nccl')
    ddp_world_size = int(os.environ['WORLD_SIZE']) 
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_rank = int(os.environ['RANK'])
    master_process = (ddp_rank == 0)
    device = f'cuda:{ddp_local_rank}'
    device_type = 'cuda'
else:
    ddp_world_size = 1
    ddp_local_rank = ddp_rank = 0
    master_process = True
    # vanilla, non-ddp run
    device = device_type = 'cpu'
    if torch.cuda.is_available():
        device = device_type = 'cuda'
    print(f'using device: {device}')


## ------------ Construct Model ------------
#  Default model setting (ref to GPT2) is as follows:
#  - @vocab_size: int = 50257  # number of tokens
#  - @block_size: int = 1024   # max sequence length
#  - @n_layer: int = 12        # number of layers
#  - @n_head: int = 12         # number of heads
#  - @n_embd: int = 768        # embedding dimension
#  - @flash: bool = False      # flag of flash attention
vocab_size = 50304  # 50304 % 128 == 0, it's better than 50257
flash_flag = True
model_config = GPTConfig(flash=flash_flag, vocab_size=vocab_size)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

model = GPT(model_config)
model.to(device)
model = torch.compile(model)

# because no gradient synchronizaton and backward in evaluate and generate
# no need to use DDP wrapped model in evaluate and generate
# so, the DDP wrapper is set up in the Train() class

## ------------ Perform pretraining ------------
def check_file_path(writing_params=None,
                    reading_params=None,
                    eval_data=None,
                    ft_data=None,):
    if writing_params:
        assert args.writing_params_path is not None, 'specify a file to save params!'
    if reading_params:
        assert args.reading_params_path is not None, 'no params file!'
    if ft_data:
        assert args.ft_data_path is not None, 'specify a FT data path!'
    if eval_data:
        assert args.eval_data_path is not None, 'specify a evaluation data path!'


if args.function == 'pretrain':
    """
    Start a new training or restart from a checkpoint.
    Careful! don't restart after all steps are finished, because
    the learning rate schedule would not match.
    """
    check_file_path(writing_params=True)

    optimizer = model.configure_optimizer(weight_decay=0.1, lr=64-4, device_type=device_type)

    #  Setting in TrainerConfig (ref to GPT2) is as follows:
    #  - lr schedule:
    #    @max_lr = 6e-4
    #    @min_lr = 0.1 * max_lr
    #    @warmup_steps = 715
    #
    #  - batch and minibatch (for gradient accumulation)
    #    @total_batch_size = 589824 # 2**16*9, ~0.5million in GPT2
    #                               # and divisible by B=24
    #    @max_steps = 19073
    #    @B = 24    # mini-batch size as large as memory can accommodate
    #    @T = 1024  # GPT2 is 1024

    train_config = TrainerConfig(total_bsize=589824, 
                                 B=24,
                                 T=1024,
                                 world_size=ddp_world_size, 
                                 rank=ddp_rank,
                                 local_rank = ddp_local_rank,
                                 device = device,
                                 device_type = device_type,
                                 master=master_process,
                                )
    ckpt_path = args.ckpt_path
    gpt_trainer = Trainer(model, optimizer, train_config, ckpt_path=ckpt_path)
    gpt_trainer.train()

    # save model
    torch.save(model.state_dict(), args.writing_params_path)


## ------------ Perform finetuning ------------
elif args.function == 'finetune':
    check_file_path(reading_params=True,
                    writing_params=True,
                    ft_data=True)    

    model.load_state_dict(torch.load(args.reading_params_path))
    ft_optimizer = model.configure_optimizer(weight_decay=0.1, lr=64-4, device_type=device_type)

    ###  need to rebuild a data loader here!  ### 
    pass

    ft_config = TrainerConfig(total_bsize=12288, # 2**12*3, divisible by B=24,(0.5m @GPT2)
                              B=24,              # as large as the GPU memory can accommodate
                              T=1024,            # 1024 @GPT2
                              nworkers=ddp_world_size, 
                              worker=ddp_rank,
                              device = device,
                              device_type = device_type,
                              master=master_process,
                              )
    ft_trainer = Trainer(model, ft_optimizer, ft_config)
    ft_trainer.train()

    # save model
    torch.save(model.state_dict(), args.writing_param_path)


## ------------ Perform evaluation ------------
elif args.function == 'evaluate':
    check_file_path(reading_params=True, eval_data=False)

    # clear the label log file
    log_dir = "../log"
    os.makedirs(log_dir, exist_ok=True)
    label_book_file = os.path.join(log_dir, f"hellaswag_eval_result.txt")
    with open(label_book_file, "w") as f: # open for writing to clear the file
        pass

    # prepare model
    model.load_state_dict(torch.load(args.reading_params_path, weights_only=True))

    acc = predict(model, 
                  device=device, device_type=device_type,
                  ddp=ddp, rank=ddp_rank, world_size=ddp_world_size,
                  hf_pretrain=False, 
                  eval_book_file=label_book_file)
    
    if master_process:
        print(f'acc: {acc:.2f}')
        

## ------------ Perform generation ------------
elif args.function == 'generate':
    check_file_path(reading_params=True)

    # prepare model and encoder
    model.load_state_dict(torch.load(args.reading_params_path, weights_only=True))
    model.eval()
    enc = tiktoken.get_encoding('gpt2')

    # set sample parameters
    n = 4
    max_length = 32
    prompt = enc.encode("hello, I'm a language model,")
    ctx_tokens = generate_samples(model, prompt, n, max_length, 
                                  rank=ddp_rank, device=device, device_type=device_type)
    
    # print the generated text row by row
    for i in range(n):
        tokens = ctx_tokens[i, :max_length].tolist()
        decode = enc.decode(tokens)
        print(f'rank {ddp_rank} sample {i}: {decode}')

#  destroy process group
if ddp:
    destroy_process_group()




