#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[ ]:


import zipfile
with zipfile.ZipFile('./flower_data_original_test.zip', 'r') as zip_ref:
    zip_ref.extractall('./flower_data/flower_data/test_2')


# In[ ]:


torch.cuda.empty_cache()


# In[1]:


# Imports here
get_ipython().run_line_magic('matplotlib', 'inline')

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


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). You can [download the data here](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip). The dataset is split into two parts, training and validation. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. If you use a pre-trained network, you'll also need to make sure the input data is resized to 224x224 pixels as required by the networks.
# 
# The validation set is used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks available from `torchvision` were trained on the ImageNet dataset where each color channel was normalized separately. For both sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.

# In[2]:


data_dir = './flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# TODO: Define your transforms for the training and validation sets
data_transforms_train = transforms.Compose([transforms.RandomResizedCrop(256),
                                            transforms.RandomRotation(30),
                                            transforms.ColorJitter(),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transforms_validation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transforms_test = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_dataset_train = datasets.ImageFolder(train_dir, transform = data_transforms_train)

image_dataset_validation = datasets.ImageFolder(valid_dir, transform = data_transforms_validation)

image_dataset_test = datasets.ImageFolder(test_dir, transform = data_transforms_validation)


# TODO: Using the image datasets and the trainforms, define the dataloaders
# define dataloader parameters
batch_size =256
num_workers=4

dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)

dataloader_valid = torch.utils.data.DataLoader(image_dataset_validation, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)

dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)


# In[4]:


class_names = image_dataset_train.classes


# In[5]:


import json

with open('cat_to_name.json', 'r') as f:
    label_map = json.load(f)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[6]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

temp = []
dictList = []
classes = []

for key, value in cat_to_name.items():
    temp = [key,value]
    dictList.append(temp)

for i in range(len(dictList)):
    dictList[i][0] = int(dictList[i][0])
    
dictList.sort()

for i in range(len(dictList)):
    classes.append(dictList[i][1])


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[7]:


#Load DenseNet201 
model= models.densenet161(pretrained=True)


# In[ ]:


for name, child in model.features.named_children():
    print (name)


# In[8]:


#Freeze denseblock layers for retraining
for name, child in model.features.named_children():
    if name in ['conv0', 'norm0','relu0','pool0','denseblock1','transition1','denseblock2','transition2','transition3','norm5']:
        print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

    else:
        print(name + ' is unfrozen')
        for param in child.parameters():
            param.requires_grad = True


# In[ ]:


model


# In[9]:


#Define final layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
class ClassifierH2(nn.Module):
    def __init__(self, inp = 2208, h1=1024, output = 102, drop=0.35):
        super().__init__()
        self.adaptivePool = nn.AdaptiveAvgPool2d((1,1))
        self.maxPool = nn.AdaptiveMaxPool2d((1,1))
        
        self.fla = Flatten()
        self.batchN0 = nn.BatchNorm1d(inp*2,eps=1e-05, momentum=0.1, affine=True)
        self.dropout0 = nn.Dropout(drop)
        self.fc1 = nn.Linear(inp*2, h1)
        self.batchN1 = nn.BatchNorm1d(h1,eps=1e-05, momentum=0.1, affine=True)
        self.dropout1 = nn.Dropout(drop)

        self.fc3 = nn.Linear(h1, output)
        
    def forward(self, x):
        adaptivePool = self.adaptivePool(x)
        maxPool = self.maxPool(x)
        x = torch.cat((adaptivePool,maxPool),dim=1)
        x = self.fla(x)
        x = self.batchN0(x)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.batchN1(x)
        x = self.dropout1(x)         
        x = self.fc3(x)
        
        return x


# In[10]:


#Set Model
model = nn.Sequential(*list(model.children())[:-1],ClassifierH2(),nn.LogSoftmax(dim=1))


# In[11]:


#Train on multiple GPUs
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


# In[12]:


# Criteria NLLLoss which is recommended with Softmax final layer
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.module.parameters(), lr=0.01)

#Define scheduler to decrease learning rate with increase in epochs
sched = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)


# In[ ]:


#Train and Evaluate the model
#Training
# number of epochs to train the model
n_epochs = 50

since = time.time()

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # track change in validation loss


train_losses, valid_losses = [], []
for epoch in range(n_epochs):
    ###################
    # train the model #
    ###################

    sched.step()
    model.train()
    train_loss = 0.0
    
    print("Starting Training")
    for data, target in dataloader_train:
        # move tensors to GPU
        data, target = data.to(device,  dtype=torch.float), target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # calculate the batch loss
        loss = criterion(output, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update training loss 
        train_loss += loss.item()
        
    else:
        valid_loss = 0.0
        accuracy = 0
        
        ######################    
        # validate the model #
        ######################
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
        print("Starting Validation")
        model.eval()
        with torch.no_grad():
            for data, target in dataloader_valid:
                # move tensors to GPU
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # update average validation loss 
                valid_loss += criterion(output, target)
                
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == target.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        train_losses.append(train_loss/len(dataloader_train))
        valid_losses.append(valid_loss/len(dataloader_valid))
        
        
        print("Epoch: {}/{}.. ".format(epoch+1, n_epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(dataloader_train)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloader_valid)),
              "Test Accuracy: {:.3f}".format(accuracy/len(dataloader_valid)))
        
        time_elapsed_total = time.time() - since
        print('Training plus Validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed_total // 60, time_elapsed_total % 60))

        # Average validation loss
        valid_loss = valid_loss / len(dataloader_valid)

        # If the validation loss is at a minimum
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            
            # Save the model
            torch.save({
                        'epoch': epoch,
                        'state_dict':model.module.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss': valid_loss,
                        'class_mapping': image_dataset_train.class_to_idx
                        },'model_D161_v1epoch{}_{:.3f}'.format(epoch,accuracy/len(dataloader_valid))+'.pth')
                       
            
            #Track model names in text file. 
            #file = open('./FileNames.txt', mode='a')
            #file.write('model_v3epoch{}_{:.3f}'.format(epoch,accuracy/len(dataloader_valid))+'.pth\n')
            #file.close()
   
            
            valid_loss_min = valid_loss


# In[ ]:


def calc_accuracy(model, cuda=False):
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader_test):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
            if idx == 0:
                print(predicted) #the predicted class
                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
            if idx == 0:
                print(equals)
            print(equals.float().mean())


# In[ ]:


calc_accuracy(model, True)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[ ]:


# TODO: Save the checkpoint 
torch.save({'epoch': epoch,
            'state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss': valid_loss,
            'class_mapping': image_dataset_train.class_to_idx
            },'model_v1_Final.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[ ]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    
    model = models.densenet201(pretrained=True)
   
    for param in model.parameters():
        param.requires_grad = False
     
    # Put the classifier on the pretrained network
    classifier = ClassifierH2()
    
    model = nn.Sequential(*list(model.children())[:-1], classifier, nn.LogSoftmax(dim=1))
    
    #Load the state dic
    model.load_state_dict(checkpoint['state_dict'],strict = False)
    
    #Load the Class Maps
    model.class_to_idx = checkpoint['class_mapping']
     
    return model

model= load_checkpoint('model_v3epoch51_0.985.pth')#'./Trained Models/model_v1epoch47_0.983.pt')


# In[ ]:


#Load the optimizer
checkpoint = torch.load('model_v3epoch0_0.965.pth')

#TODO Define Checkpoint
optimizer = optim.Adam(model.module.parameters(), lr=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#Load the start epoch
#TODO Define Checkpoint
start_epoch = checkpoint['epoch']

criterion = nn.NLLLoss()

sched = lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[ ]:


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[ ]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[ ]:


image_path = 'flower_data/flower_data/test/1/1.jpg'
img = process_image(image_path)
imshow(img)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[ ]:


def predict(image_path, model, top_num=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
        # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_map[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the validation accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:


# TODO: Display an image along with the top 5 classes
def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
    flower_num = image_path.split('/')[-2]
    title_ = label_map[flower_num]
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()


# In[ ]:


model.eval()
image_path = './flower_data/flower_data/test/30/5.jpg'
plot_solution(image_path, model)


# In[ ]:


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path, map_location ='cpu')
    
    model = models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
     
    # Put the classifier on the pretrained network
    classifier = ClassifierH2()
    
    model = nn.Sequential(*list(model.children())[:-1], classifier, nn.LogSoftmax(dim=1))
    
    #Load the state dic
    model.load_state_dict(checkpoint['state_dict'],strict = False)
    
    return model

model= load_checkpoint('./model_D121_v1epoch0_0.160.pth')

