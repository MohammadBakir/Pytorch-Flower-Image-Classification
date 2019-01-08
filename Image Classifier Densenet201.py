#Importing required Packages

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


## Load the data

''' Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). 
You can [download the data here](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip). 
The dataset is split into two parts, training and validation. For the training, you'll want to apply transformations 
such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. 
If you use a pre-trained network, you'll also need to make sure the input data is resized to 224x224 pixels as required
by the networks. The validation set is used to measure the model's performance on data it hasn't seen yet. 
For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the 
appropriate size. The pre-trained networks available from `torchvision` were trained on the ImageNet dataset where each 
color channel was normalized separately. For both sets you'll need to normalize the means and standard deviations of the 
images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the 
standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel 
to be centered at 0 and range from -1 to 1.'''

#Defining data directories, modify accordingly
data_dir = './flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define transforms for the training and validation sets
#Using pretrained Pytorch model trained on image sizes 224. Modify ResizedCrop and CenterCrop according to needs. 
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

#Load the datasets with ImageFolder
image_dataset_train = datasets.ImageFolder(train_dir, transform = data_transforms_train)

image_dataset_validation = datasets.ImageFolder(valid_dir, transform = data_transforms_validation)

image_dataset_test = datasets.ImageFolder(test_dir, transform = data_transforms_validation)


# Using the image datasets and the trainforms, define the dataloaders
#batch size and num workers can be modified accordingly. 
batch_size =256
num_workers=4
 

dataloader_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)

dataloader_valid = torch.utils.data.DataLoader(image_dataset_validation, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)

dataloader_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True)


#Define classnames
class_names = image_dataset_train.classes


# ### Label mapping
'''You'll also need to load in a mapping from category label to category name. 
You can find this in the file `cat_to_name.json`. 
It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). 
This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.'''
import json

with open('cat_to_name.json', 'r') as f:
    label_map = json.load(f)


# # Building and training the classifier
# 
''' Now that the data is ready, it's time to build and train the classifier. Will be using a pretrain Pytorch model from
'torchvision.models' to get image features. Model will be built and trained with a new feed-forward classifier using those features '''


# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) 
#Load DenseNet201 
model= models.densenet201(pretrained=True)

#Freeze denseblock layers for retraining, Optional
for name, child in model.features.named_children():
    if name in ['conv0', 'norm0','relu0','pool0','denseblock1','transition1','denseblock2','transition2','transition3','norm5']:
        print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

    else:
        print(name + ' is unfrozen')
        for param in child.parameters():
            param.requires_grad = True

            
# * Define a new, untrained feed-forward network as a classifier
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
      
class ClassifierH2(nn.Module):
    def __init__(self, inp = 1920, h1=1024, output = 102, drop=0.35):
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

#Set Model
model = nn.Sequential(*list(model.children())[:-1],ClassifierH2(),nn.LogSoftmax(dim=1))


#Train on multiple GPUs if available, else train on cpu
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)


#Define optmizer and scheduler
# Criteria NLLLoss which is recommended with Softmax final layer
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
#Note,  model.module.parameters() is referencing the DataParallel module. If model is trained on CPU or individual GPU will need to be modifed Accordingly. 

#according to how model was trained. 
optimizer = optim.Adam(model.module.parameters(), lr=0.01)

#Define scheduler to decrease learning rate with increase in epochs
sched = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)


#Train and Evaluate the model
#Training
#Track the loss and accuracy on the validation set to determine the best hyperparameters

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
            #Note, state_dict is saved as model.module.state_dict() as this model was trained using DataParallel. Will need to be modified
            #according to how model was setup. 
            torch.save({
                        'epoch': epoch,                            
                        'state_dict':model.module.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss': valid_loss,
                        'class_mapping': image_dataset_train.class_to_idx
                        },'model_D201_v1.pth')
                                  
            valid_loss_min = valid_loss



#Function to test model accuracy against test data. 
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

            
calc_accuracy(model, True)


# ## Loading the checkpoint
# At this point it's good to write a function that can load a checkpoint and rebuild the model. 
#That way you can come back to this project and keep working on it without having to retrain the network.


# Function that loads a checkpoint and rebuilds the model
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

#Model name as defined during training run
model= load_checkpoint('model_D201_v1.pth')


#To load optimizer state
checkpoint = torch.load('model_D201_v1.pth')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# # Inference for classification

'''The below functions are used to train the network for inference. An image will be passed into the network 
and predict the class of the flower in the image.''' 


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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. 
# If your `process_image` function works, running the output through this function should return the 
# original image (except for the cropped out portions).

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

#Test out above functions.
image_path = 'flower_data/flower_data/test/1/1.jpg'
img = process_image(image_path)
imshow(img)


# ## Class Prediction
#Function for making predictions with your model. 

def predict(image_path, model, top_num=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
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
# Display an image along with the top 5 classes
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


model.eval()
image_path = './flower_data/flower_data/test/30/5.jpg'
plot_solution(image_path, model)
