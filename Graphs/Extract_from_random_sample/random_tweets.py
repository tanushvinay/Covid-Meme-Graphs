# -*- coding: utf-8 -*-
"""random.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s6l972dd5qRyt0mCL9X-_iWDaUj5DDE_
"""

import PIL
from PIL import Image
import tensorflow as tf
from torchvision import transforms
from torchvision import models
import torch
import torch.autograd as Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import pandas as pd
from pandas import ExcelFile
import requests
from io import BytesIO
from tqdm import tqdm
from glob import glob
import json

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
        x = self.Lin1(x)
        x = self.Lin2(x)
        x = self.Lin3(x)
       
        
        return F.log_softmax(x, dim = 1)

def extraction(file):
  count = 0
  all_images, meaningful_images = [], {}
  for line in tqdm(open(file,"r").read().split("\n")):
    count += 1
    try:
      result = json.loads(line)
      if isinstance(result['text'], str) and str(result).find('media_url') != -1:
        string = str(result)
        start = string.find('media_url')
        link = string[string.find('http',start) : string.find('jpg',start)+3]
        all_images.append(link)
        matches = ["graph", "graphic","chart", "diagram", 
                  "venn", "pie", "historgram", 
                  "visualization", "visualisation","graphical", 
                  "representation", "statistc",  "data", "map",
                  "figure", "axis", "axes", "plot", "cartograph", "analysis"]
        if any(x in str(result['text']).lower() for x in matches):
          meaningful_images[result['id_str']] = link

    except Exception as e:
      pass
      #print(e)
  return all_images, meaningful_images

def check_cuda():
  _cuda = False
  if torch.cuda.is_available():
    _cuda = True
  return _cuda

def test_img(model, input):
  try:
    response = requests.get(input)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    simple_transform = transforms.Compose([transforms.Resize((512,512)),
                              transforms.ToTensor(),
                              transforms.Normalize([0.458, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
    inp = simple_transform(img)[None, ...]
    inp = inp.cuda()
    output = model(inp)

    indices, value = torch.max(output.data, 1)
    out = [output, indices, value]
  except Exception as e:
    out = None 
  return out

def classify(dictionary, model):
  classes = ['Bar', 'Bubble', 'Line', 'Map']
  for link in tqdm(dictionary):
    output = test_img(model, dictionary[link])
    if output == None: continue
    class_prob = output[1].detach().cpu().numpy()
    pred = output[2].detach().cpu().numpy()
    prediction = classes[pred[0]]
    if class_prob < -0.03:
      continue
    response = requests.get(dictionary[link])
    filename = str(link)
    img = Image.open(BytesIO(response.content)).convert('RGB').save('new_graphs/' + filename + '.jpeg')

model = torch.load('graphs_1200')
is_cuda = False
if is_cuda:
  model.cuda()
try:
  all_images, meaningful_images = extraction("random_tweets.jsonl")
  classify(meaningful_images, model)
except Exception as e:
  print(e)


links_df = pd.DataFrame()
links_df['links'] = all_images
links_df.to_csv('all_urls.csv')
