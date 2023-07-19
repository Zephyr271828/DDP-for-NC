import argparse
import os
import random
#from m_cifar10 import train, analysis, graphs, plot
from socket import gethostname

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from ddp_utils import *

# You may add this to the .slurm file
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --partition=v100
#SBATCH --partition=a100_1
#SBATCH --partition=a100_2

# =================================================================
# A preliminary model
# =================================================================
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12349"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size):

    # Hyperparameters
    epochs = 350
    lr = 0.1
    lr_decay = 0.7
    epochs_lr_decay = [epochs//3, epochs*2//3]
    epoch_list = [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,   11,
                  12,  13,  14,  16,  17,  19,  20,  22,  24,  27,   29,
                  32,  35,  38,  42,  45,  50,  54,  59,  65,  71,   77,
                  85,  92,  101, 110, 121, 132, 144, 158, 172, 188,  206,
                  25, 245, 268, 293, 320, 350]

    batch_size = 128
    momentum = 0.9
    weight_decay = 5e-4

    dataset = datasets.CIFAR10
    model = models.resnet18
    im_size             = 28
    padded_im_size      = 32
    C                   = 10
    input_ch            = 3
    loss_name = 'CrossEntropyLoss'
    #loss_name = 'MSELoss'
    optimizer = optim.SGD

    save_model = True
    debug = True
    log_interval = 10

    # set the seed manually
    SEED = 2023
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # ddp setup
    ddp_setup(rank, world_size)
    local_rank = rank - world_size * (rank // world_size)
    torch.cuda.set_device(local_rank)

    # Model
    model = model(weights = None, num_classes = C)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    model = model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # Loss Function
    if loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif loss_name == 'MSELoss':
        criterion = nn.MSELoss()
        criterion_summed = nn.MSELoss(reduction='sum')
    
    # Optimizer
    optimizer = optimizer(ddp_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=epochs_lr_decay, gamma=lr_decay)
    
    # Dataset
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size)//2),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])
    train_set = dataset('data', train=True, download=False, transform=transform)
    test_set = dataset('data', train=False, download=False, transform=transform) 

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank))

    analysis_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank))

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(test_set, num_replicas=world_size, rank=rank))

    # graphs
    train_graphs = graphs()
    test_graphs = graphs()
    if dataset == datasets.CIFAR10: no = 0
    elif dataset == datasets.CIFAR100: no = 1
    elif dataset == datasets.FashionMNIST: no = 2
    elif dataset == datasets.MNIST: no = 3
    else: no = 4
    no = 4

    # Main body
    cur_epochs = []
    for epoch in range(1, epochs + 1):
        #train_sampler.set_epoch(epoch) #shuffle for DDP
        train(ddp_model, criterion, local_rank, C, train_loader, optimizer, epoch, batch_size, debug)
        print(f"epoch {epoch} training finished")
        lr_scheduler.step()

        if epoch in epoch_list:
            cur_epochs.append(epoch)
            analysis(train_graphs, model, ddp_model, criterion_summed, local_rank, C, analysis_loader, weight_decay, epoch, debug, loss_name)
            analysis(test_graphs, model, ddp_model, criterion_summed, local_rank, C, test_loader, weight_decay, epoch, debug, loss_name)
            print(f"epoch {epoch} analysis finished")

        if rank == 0 and (epoch in epoch_list[::7] or epoch in epoch_list[41::2]):
        #if rank == 0 and epoch in epoch_list:
            plot(epoch, cur_epochs, train_graphs, test_graphs, loss_name, no)
            print(f"epoch {epoch} plotting finished")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args = (world_size,), nprocs = world_size)
