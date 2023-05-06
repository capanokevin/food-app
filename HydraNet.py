import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import random
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from collections import defaultdict
from recipe_encoder import * 
from image_encoder import *
from utils.preprocessing import get_image_loader, get_recipe_loader, recipe_preprocessing 
from utils.utils import load_checkpoint, count_parameters
import shutil

import torchvision.models as models
from collections import OrderedDict



class Hydra(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.mobilenet_v2(pretrained = True)  
        self.net.features[0][0].in_channels = 4 
        xavier_tensor = torch.empty(32, 4, 3, 3, requires_grad=True)
        nn.init.xavier_uniform_(xavier_tensor)
        self.net.features[0][0].weight = nn.Parameter(xavier_tensor)
        self.net.classifier[1] = nn.Linear(1280, 1280)

        self.n_features  = self.net.classifier[1].in_features
        self.net.fc = nn.Identity()

        self.net.fc1 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features, 1024)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.6)),
            ('final', nn.Linear(1024, 1))]))

        self.net.fc2 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,1024)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.6)),
            ('final', nn.Linear(1024, 1))]))

        self.net.fc3 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,1024)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.6)),
            ('final', nn.Linear(1024, 1))]))
        
        self.net.fc4 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,1024)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.6)),
            ('final', nn.Linear(1024, 1))]))

        self.net.fc5 = nn.Sequential(OrderedDict(
            [('linear', nn.Linear(self.n_features,1024)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(0.6)),
            ('final', nn.Linear(1024, 1))]))
        


        
    def forward(self, x):#, depth):
        # Freeze the first 10 layers
        #for param in self.net.features[:18].parameters():
        #    param.requires_grad = False

        x = self.net(x)
        
        kcal_head = self.net.fc1(x)
        carbo_head = self.net.fc2(x)
        protein_head = self.net.fc3(x)
        fat_head = self.net.fc4(x)
        mass_head = self.net.fc5(x)

        return kcal_head, carbo_head, protein_head, fat_head, mass_head