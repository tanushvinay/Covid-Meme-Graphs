import os
import shutil
import pprint 

from glob import glob
import numpy as np
#import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch
import torch.autograd as Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tqdm import tqdm

## CNN class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 8, 3), # inp (3, 512, 512)
            nn.Conv2d(8, 16, 3),
            nn.MaxPool2d(2),
            nn.ReLU() # op (16, 256, 256)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5), # inp (16, 256, 256)
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU() # op (32, 64, 64)
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), # inp (32, 64, 64)
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU() # op (64, 32, 32)
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(64, 128, 5), # inp (64, 32, 32)
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU() # op (128, 16, 16)
        )
        self.Lin1 = nn.Linear(15488, 1500)
        self.Lin2 = nn.Linear(1500, 150)
        self.Lin3 = nn.Linear(150, 4)
        
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        #x = torch.flatten(x, 1)
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
       
        
        return F.log_softmax(x, dim = 1)

#!rm -rf dataset/train/.DS_Store
#!rm -rf dataset/valid/.DS_Store

def rm_r(path):
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

try:
    rm_r('dataset/train/.DS_Store')
    rm_r('dataset/valid/.DS_Store')
except:
    pass

simple_transform = transforms.Compose([transforms.Resize((512,512)),
                           transforms.ToTensor(),
                           transforms.Normalize([0.458, 0.456, 0.406],
                           [0.229, 0.224, 0.225])])
train = ImageFolder('dataset/train/', simple_transform)
valid = ImageFolder('dataset/valid/', simple_transform)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_data_loader = DataLoader(train, batch_size = 16, num_workers = 3, shuffle=True)
valid_data_loader = DataLoader(valid, batch_size = 16, num_workers = 3, shuffle=True)

def check_cuda():
  _cuda = False
  if torch.cuda.is_available():
    _cuda = True
  return _cuda

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
with strategy.scope():

  model = Net()
  is_cuda = check_cuda()
  print(is_cuda)
  if is_cuda:
    model.cuda()

  '''model = torch.load('graphs')
  model = model.train()
  is_cuda = check_cuda()
  if is_cuda:
    model.cuda()'''

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)
  for epoch in tqdm(range(1200)):
    running_loss = 0.0 
    for i,data in enumerate(train_data_loader, 0):
      
      inputs, labels = data
      inputs, labels = inputs.cuda(), labels.cuda()
      optimizer.zero_grad()

      #forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

print('Finished Training')
torch.save(model, 'graphs_1200')

model = torch.load('graphs_1200')
model = model.train()
is_cuda = check_cuda()
if is_cuda:
  model.cuda()
correct, total = 0,0
with torch.no_grad():
  for data in tqdm(valid_data_loader):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on train images: %d %%' % (
    100 * correct/total))