import torch
import sys
from gpt_model import GPT, GPTConfig

## construct a model for sanity check
# set seeds
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# set device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"using device: {device}")

# model configuration
config = GPTConfig()

# create model
model = GPT(config)
model.to(device)

## 1. sanity check for weight init
# load text
import tiktoken
enc = tiktoken.get_encoding("gpt2")
with open('../data/input.txt', 'r') as f:
    text = f.read()
    
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32 # B: Batch size, T: sequence length

# take data for constructing one batch of (input, output) pair
batch = torch.tensor(tokens[:B*T + 1])
batch = batch.to(device) # 注意这里和model不同，这里不能直接用 batch.to(device)，它的返回值才是device上新建的对象
x = batch[:-1].view(B, T)
y = batch[1:].view(B, T)

scores, loss = model(x, y)
print(loss) # sanity chekc方式：用loss值来判断weights的初始值是否合理
            # 这里vocab size = 50257，weights初始值应该让这些值的Probability均匀分布
            # 这时候的loss应该接近-ln(1/50257)，大约10.8249

# sys.exit()

# parameters setting in opt is the same as those in GPT-3
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# 在一个batch上train，看能不能overfit，确认可以后再正式train
iter_num = 50
for i in range(iter_num):
    optimizer.zero_grad() # 一定不能忘记
    scores, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step {i}, loss: {loss.item()}') # 测试通过，后面开始正式train
