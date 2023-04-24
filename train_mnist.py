import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(args):
    torch.manual_seed(args.seed)

    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url)

    if args.device == 'cuda':
        torch.cuda.set_device(args.local_rank)
        args.device = f'cuda:{args.local_rank}'


    model = ConvNet()
    model.to(args.device)

    batch_size = args.batch_size
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank], output_device=args.local_rank)
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=args.rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 :
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                                         args.epochs,
                                                                         i + 1,
                                                                         total_step,
                                                                         loss.item()))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int,
                        help='Integer seed for initializing random number generators')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to train on')

    parser.add_argument('--epochs', default=10, type=int,
                        help='Integer epochs for training')

    parser.add_argument('--batch_size', default=100, type=int,
                        help='Integer batch size for training')

    parser.add_argument('--world_size', default=1, type=int,
                        help='Total number of distributed processes (DO NOT SET)')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='The index of this process within this machine\'s training processes (DO NOT SET)')
    parser.add_argument('--rank', default=0, type=int,
                        help='The index of this process within all training processes (DO NOT SET)')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='Master URL for distributed training')
    parser.add_argument('--dist-backend', default='nccl', choices=['nccl', 'gloo', 'mpi'],
                        help='Distributed backend')

    #mp.spawn(train, nprocs=args.gpus, args=(args,))

    args = parser.parse_args()

    train(args)
