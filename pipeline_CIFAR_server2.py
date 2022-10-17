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
LEARNING_RATE = 0.05
MOMENTUM = 0.5
NUM_CHUNK = 2

nblocks = [6,12,24,16]
growth_rate = 12

MASTER_ADDR = 'localhost'
PORT = 24500

# socket server 2

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.connect((MASTER_ADDR, PORT))

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

class Shard2(nn.Module):
    def __init__(self):
        super(Shard2, self).__init__()

        block = Bottleneck
        reduction = 0.5
        num_classes = 10

        self.growth_rate = growth_rate
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(f"Using device: {torch.cuda.get_device_name(self.device)}")

        num_planes = 8*growth_rate
    
        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2]).to(self.device)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes).to(self.device)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3]).to(self.device)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes).to(self.device)
        self.linear = nn.Linear(num_planes, num_classes).to(self.device)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = self.trans3(self.dense3(x))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out.cpu()

model = Shard2()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

all_package = []

def listening_port(s):
    while True:
        package_data = recv_msg(s)
        package = pickle.loads(package_data)
        all_package.append(package)

def processing_package():
    global all_package
    count = 0
    count2 = 0
    inputs = [0]*NUM_CHUNK
    while True:
        if len(all_package) == 0: continue
        else:
            # print(len(all_package))
            recv = all_package[0]
            all_package.pop(0)
            if recv[0] == '1':
                model.train()
                inputs[recv[2]] = recv[1]
                output = model(inputs[recv[2]])
                data_string = pickle.dumps(['2',output,recv[2]])
                send_msg(s, data_string)
            elif recv[0] == '3':
                model.train()
                optimizer.zero_grad()
                out_grad = recv[1]
                count += 1
                output = model(inputs[recv[2]])
                output.backward(gradient=out_grad)
                optimizer.step()
                b = inputs[recv[2]].grad
                data_string = pickle.dumps(['4',b,recv[2]])
                send_msg(s, data_string)
            elif recv[0] == '5':
                model.eval()
                input = recv[1]
                output = model(input)
                data_string = pickle.dumps(['6',output])
                send_msg(s, data_string)
            elif recv[0] == '0':
                break

thr.Thread(target=listening_port,args=(s,)).start()
thr.Thread(target=processing_package,args=()).start()
