Pytorch-Flower-Image-Classification

## Developing an AI application

In this project, you'll train an image classifier to recognize different species of flowers. ation. 

We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories. 

## The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

Part 1: Importing Packages and Defining Datatransforms
#importing the packages you'll need.
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler


import matplotlib.pyplot as plt
import numpy as np

import helper
import zipfile
from PIL import Image

import time
import seaborn as sns

