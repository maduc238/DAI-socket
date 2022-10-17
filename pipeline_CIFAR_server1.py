import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models.resnet import Bottleneck

import os
import time
import argparse
import socket
import pickle
import struct
from tqdm import tqdm
import threading as thr
import torchvision
import torchvision.transforms as transforms
import math

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 0.05
MOMENTUM = 0.5
NUM_CHUNK = 2

nblocks = [6,12,24,16]
growth_rate = 12

# socket server

MASTER_ADDR = 'localhost'
PORT = 24500

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((MASTER_ADDR, PORT))
print('Socket binded to port', PORT)

s.listen(5)
print('Socket is listening')

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Shard1(nn.Module):
    def __init__(self):
        super(Shard1, self).__init__()

        block = Bottleneck
        reduction = 0.5
        num_classes = 10

        self.growth_rate = growth_rate
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        num_planes = 2*growth_rate

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False).to(self.device)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0]).to(self.device)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes).to(self.device)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1]).to(self.device)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes).to(self.device)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        return out.cpu()

model = Shard1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

all_package = []
c,addr = s.accept()

def listening_port(s):
    while True:
        package_data = recv_msg(s)
        package = pickle.loads(package_data)
        all_package.append(package)

def train(c):
    global all_package
    for epoch in range(1, EPOCHS + 1):
        print('Start epoch',epoch)
        model.train()
        test_loss = 0
        correct = 0
        for data, target in tqdm(train_loader):
            # variables
            count = 0
            count2 = 0
            a_grad = [0]*NUM_CHUNK
            all_data = data.chunk(NUM_CHUNK,axis=0)
            # send micro batches
            for itr, micro_data in enumerate(all_data):
                output = model(micro_data)
                data_string = pickle.dumps(['1',output,itr])
                send_msg(c, data_string)

            while True:
                if len(all_package) == 0: continue
                else:
                    # print(len(all_package))
                    recv = all_package[0]
                    all_package.pop(0)
                    if recv[0] == '2':
                        output2 = recv[1]
                        # define loss function
                        # print(output2)
                        loss = criterion(output2, target.chunk(NUM_CHUNK,dim=0)[recv[2]])
                        loss.backward()
                        a_grad[recv[2]] = output2.grad
                        count += 1
                        if count == NUM_CHUNK:
                            for i, a in enumerate(a_grad[::-1]):
                                data_string = pickle.dumps(['3',a,NUM_CHUNK-i-1])
                                send_msg(c, data_string)

                    elif recv[0] == '4':
                        optimizer.zero_grad()
                        output = model(all_data[recv[2]])
                        output.backward(gradient=recv[1])
                        optimizer.step()
                        count2 += 1
                        if count2 == NUM_CHUNK:
                            break
                    else:
                        print('some error')
                        return
        # test
        model.eval()
        for data, target in test_loader:
            output = model(data)
            data_string = pickle.dumps(['5',output])
            send_msg(c, data_string)
            while True:
                if len(all_package) == 0: continue
                else:
                    recv = all_package[0]
                    all_package.pop(0)
                    if recv[0] != '6':
                        print('failed')
                        return
                    test_loss += F.nll_loss(recv[1], target, size_average=False).data # sum up batch loss
                    pred = recv[1].data.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    break

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset)))

    data_string = pickle.dumps(['0','end'])
    print('send end')
    # send_msg(c, data_string)
    c.close()

thr.Thread(target=listening_port,args=(c,)).start()
thr.Thread(target=train, args=(c,)).start()
