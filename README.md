# India-s-best-chance

![Wee-Scope](https://i.imgur.com/99j4ggE.png)

Wee-Scope is a cloud-enabled, mobile-ready, offline-storage, Machine Learning powered Bio-Tech Product that is:

  - Cheaper than Present Alternatives
  - Light-Weight and Portable
  - High Resolution with 1000x Magnification.
  - AI Ready.

# Origin of Wee-Scope

Wee means micro. How did we come upon the idea of Making a Wee-Scope?
It all started with us visiting a veevor village in a NSS camp. We found that couple of children were suffering from diseases and they didn't know what they were suffering from. Upon asking why, we came to know two things.
- There was no proper facilities or doctors available in the village at all times to diagonose advanced diseases.
- Even if they were available the economic conditions of the villagers didn't allow them to avail those Healthcare facilities which they deserved.

Healthcare should be the right of Every Human being. Healthcare isn't a national issue, but a Humanitarian Issue. Our study found that nearly 8 million people die every year because of a lack of access to high-quality care. That is a big problem for 21st century where everyone cares about money and fairy tales of eternal economic growth and going to Mars. 

# Why Wee-Scope?

Why Wee-Scope? Because ours is a AI Ready, Portable and most Importantly a Lot Cheaper than traditional Disease Diagonising Costs, i.e., Cost for diagonising the disease in a Pathology Laboratory, or being able to afford Disease diagonising machines. Not to mention the middle man costs, Such as paying to pathology laboratories and Doctors. 
![Patho_Cost](https://i.imgur.com/KH2kTRT.png)
Also the factor of time. When you give your blood to test it takes usually 2-7 days for your results to come back. Our system would cut all the redundant and unwanted cost plus would cut the time required in diagonising the disease. Our system will enable anyone using it to diagonise a disease in mere 15 minutes. That would cut the time required for diagonizing by 1/8th.

# How did we make it, and How much does it cost?

The first iteration of idea came from Manu Prakash's foldscope, as good as it sounded to us the lower resolution and fixation of lens lead to chaos. The basic idea of foldscope is combining 2.6 mm lens to get a magnification of about 140x, however the resolution and magnification was very low even to identify basic diseases.
![foldscope_blood](https://i.imgur.com/DDBlPUK.jpg)
The second iteration of wee-scope followed by re-desiging the microscope which prove to be expensive than existing one.
But our main focus while making Wee-Scope was to keep the quality of the product durable, Price of the product cheap and accurate which would work as effectively as a traditional microscope. 
It can be made from something as cheap as a thick paper and something as hard as metal pipe. For Demonstration purposes, we made the wee-scope using the cardboard packing of the the Jet Brain Bottle Goodie. We even made a Wee-Scope out of the cover of the notepad given by the Hack-A-Bit.
![wee-scope-paper](https://i.imgur.com/gfSQnaV.jpg)
Our Wee-Scope consist of a tubular structure that can be made out of almost anything you can find your hands on, something as cheap as paper and something as lightweight and strong as a PVC Pipe. At one side of the tube is attached a 10x objective lens which is 2.0 mm in diameter. And another end consists of a 100x spherical Objective. 10x objective multiplied with 100x objective at a specific focal length gives us 1000x Magnification. After much trial and error we found out the perfect Focal length for this two objectives was 20 cms. Our design diagram can be seen below.
![wee-scope-diagram](https://i.imgur.com/E7dEqnI.jpg?1)

The slide housing was created from old shoe box (cardboard) by stacking the cutted cardboard on one another and making an entry point for slide.

![cardboard-dig](https://i.imgur.com/1SuMXsl.jpg)

![cardboard-made](https://i.imgur.com/D2Szeyj.jpg)

# Process and cost for Malaria diagnosis (Example)

Traditionally, when patient is suspected to be diagnosed with malaria doctor collects the blood of the patient and sends it for malaria diagnosis. However, this process requires time and more human resources to process a single report (2 days). After desired number of days patient is adviced with the result, this increases cost of diagnosis since more human resource is involved.

However, Wee-scope is able to diagnose malaria instantly within minutes with the help of machine learning which in retrospect decreases the human resource therby cutting the cost. Here is the cost of malaria dignosis kit provided by Wee-scope

Giemsa Stain - 100ml  (9ml is required to process 15 slides, therefore 1500/9 = 167 Patients | Rs 278)
Slides - 167 slides (One slide = Rs2 | 167 x 2 = Rs334)
Lancets - 167 (167 x 0.6 = Rs 100)

Total = Rs 712 | For 167 Patients
Avg. Time for report generation = ~15 minutes (For single patient)

-----------------------------------------------------------------------------------------------------------------

Traditional method

Malaria Parasite test = Rs 130/Patient | 130 x 167 = Rs 21710
Avg. Time for report generation= ~48 hours (For single patient)

-----------------------------------------------------------------------------------------------------------------
Comparing the above statistics

Wee-scope for 167 patient = Rs 712 | with a single kit
Traditional test for 167 patients = Rs 21710
Therefore,
21710/712 = 30.5

Wee-scope is 30 times cheaper than traditional method of diagnosis of malaria.

# Life in wee-scope

Wee-scope is cheap, light, portable and as effective as microscope in real life. Wee-scope provides magnification upto 1000x enough to diagnose diseases like sickle cell anemia with bare eyes. Wee-scope was tested with plasmodium slides acquired from local pharmacy college to test the resolution and quality of magnification. Given below are the few example of real images taken from wee scope of malaria infected blood smear.
The purple color dots you see in the below images are malaria parasites.

![blood-in-wee](https://i.imgur.com/UbjClCn.png)

![blood-in-wee2](https://i.imgur.com/4yUqUPk.jpg)

# SOFTWARE IMPLEMENTATION
# Back-End

We are implementing our Back end on PyTorch and we are deploying it on Google Colab.

`!wget https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip`

To train our machine we used the dataset of 27,558 images. This dataset contains two folders, one containing infected samples and the other contains the uninfected ones. National Library of Medicine has provided this dataset. 

`!unzip cell_images.zip`

Since the dataset is downloaded in Zip format We use this command to unzip the dataset.

```
import numpy as np
import pandas as pd
import os
import PIL
import torch
import torchvision
from matplotlib.pyplot import *
import skimage.io

root_path = "/content/cell_images"
uninfected_cell_images = os.listdir(root_path + "/Uninfected")
infected_cell_images = os.listdir(root_path + "/Parasitized")
healthy_im = skimage.io.imread(os.path.join(root_path, "Uninfected", uninfected_cell_images[0]))
unhealthy_im = skimage.io.imread(os.path.join(root_path, "Parasitized", infected_cell_images[0]))

figure(figsize=(15, 10))

subplot(1, 2, 1)
title("Healthy")
imshow(healthy_im)
xticks([]);yticks([]);

subplot(1, 2, 2)
title("Unhealthy")
imshow(unhealthy_im)
xticks([]);yticks([]);

```
Here we import all the necessary modules requried such as numpy and matplotlib. This section labels the images in datasets and segregates them into Healthy sample and Unhealthy sample.

```
import seaborn as sn

sizes = np.zeros((20, 20), dtype=np.dtype(int))
for healthy_im_fname in uninfected_cell_images:
    if healthy_im_fname.endswith(".png"):
        w, h = PIL.Image.open(os.path.join(root_path, "Uninfected", healthy_im_fname)).size
        w = min(190, w)
        h = min(190, h)
        sizes[w // 10, h // 10] += 1

for sick_im_fname in infected_cell_images:
    if sick_im_fname.endswith(".png"):
        w, h = PIL.Image.open(os.path.join(root_path, "Parasitized", sick_im_fname)).size
        w = min(190, w)
        h = min(190, h)
        sizes[w // 10, h // 10] += 1

figure(figsize=(20,15))
df = pd.DataFrame(sizes)
sn.heatmap(df, fmt='d', annot=True)
xlabel("Width of image (in 10 pixel increments)")
ylabel("Height of image (in 10 pixel increments)")
title("Image size and shape distribution.")

```
What the above code does is create a table showing the distribution of image shapes / sizes.

```
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random


data_transforms = {
    'test': transforms.Compose([transforms.Resize((64, 64)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(5),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'train': transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


train_set = torchvision.datasets.ImageFolder(root_path, transform=data_transforms['train'])
test_set = torchvision.datasets.ImageFolder(root_path, transform=data_transforms['test'])

num_train = len(train_set)
indices = list(range(num_train))
np.random.shuffle(indices)

valid_size = 0.2 # 20% of images used for validation
test_size = 0.1 # 10% of images used for testing.

start_valid = int(num_train - (num_train * valid_size + test_size))
start_test = int(num_train - (num_train * test_size))

train_sampler = SubsetRandomSampler(indices[0:start_valid])
val_sampler = SubsetRandomSampler(indices[start_valid:start_test])
test_sampler = SubsetRandomSampler(indices[start_test:])
```

Two data sets are created, namely test and train these images are resized to standard size 64x64 and RGB values are normalized to 0.5

```
def togpu(x):
    return x.cuda()
def tocpu(x):
    return x.cpu()
```
GPU and CPU are set up for training, namely CUDA library of Nvidia.

```
from torch.autograd import Variable
import torch.nn.functional as F

# Now we get to the CNN
class MalariaCNN(torch.nn.Module):
    
    def __init__(self):
        super(MalariaCNN, self).__init__()
        
        # The shape of the images are 64 x 64 x 3
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        # After pooling the images should be 16 x 16
        self.fc1 = torch.nn.Linear(in_features=(64 * 16 * 16), out_features=1024)
        self.fc2 = torch.nn.Linear(in_features=1024, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=2)
        
    def forward(self, x):
        
       
        x = F.relu(self.conv1(x))
        
        
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        
       
        x = F.max_pool2d(x, 2)
        
        x = x.reshape(-1, 64 * 16 * 16)
        
       
        x = F.relu(self.fc1(x))
        
       
        x = F.relu(self.fc2(x))
        
     
        x = self.fc3(x)
        
        return x
 ```
 Convulation is performed here, in conv1 size changes from (3, 64, 64) to (64, 64, 64)
 
 ```
 

def get_train_loader(batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    
    return train_loader

test_loader = DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

import torch.optim as optim

def createLossAndOptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)
```
Data header takes in data set for loading.

```
import time

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    training_start_time = time.time()
    
    min_val_loss = 1.0
    
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 5 # Print 10 times total each epoch
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            
            #print(inputs, labels)
            inputs, labels = Variable(togpu(inputs)), Variable(togpu(labels))
            
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            # print(loss_size.data.item())
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            
            if i % print_every == print_every - 1:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time
                ))
                
                running_loss = 0.0
                start_time = time.time()
                
        total_val_loss = 0
        for inputs, labels in val_loader:
            inputs, labels = Variable(togpu(inputs)), Variable(togpu(labels))
            
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        if (total_val_loss / len(val_loader)) < min_val_loss:
            print("New best: ({} -> {})".format(min_val_loss, total_val_loss / len(val_loader)))
            min_val_loss = total_val_loss / len(val_loader)
            torch.save(net.state_dict(), "malaria_best.pt")
                                
        
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
```
Pretraining attributes are set that includes label and data.

```
CNN = togpu(MalariaCNN())
trainNet(CNN, batch_size=64, n_epochs=25, learning_rate=0.002)
```
Training batch is deployed using CNN model for 25 iterations with a batch size of 64.

__________________________________________________________________________________________________
# Deploying CNN model to webapp.

Anvil was used to deploy trained model to webapp, But what is Anvil? Well, it is a tool to create a web interface for any Python project, using pure Python. No HTML, CSS or JavaScript is required. Images that were used to diagnose disease were saved at img.save, however the backup was saved at /backup folder for further training of current model which is triggered and corrected every week to make the model better and more accurate.

```
pip install anvil-uplink
```


```
import anvil.server
anvil.server.connect('ZDE4xxxxxxxxxxxxxxxxxxxxxxxxxxR')
import anvil.media
import socket
import io
import os
from random import randint
from PIL import Image


@anvil.server.callable
def classify_image(file):
    rand = random.Random()
    
    with anvil.media.TempFile(file) as filename:
      img = Image.open(io.BytesIO(file.get_bytes()))
      bs = io.BytesIO()
      #img.save(bs, format="JPEG")
      # save image in upload 
      img.save('/content/cell_images/upload/myphoto.png', 'PNG')
      final_path = '/content/cell_images/upload/myphoto.png'
      prediction = "Sick" if getPrediction(CNN, final_path) == 0 else "Healthy"
      # save image for backup      
      lol = str(randint(1,1000000))
      backup_path = "/content/cell_images/backup/"+lol+".png"
      img.save(backup_path, 'JPEG')
      return(prediction)
try:
  os.remove('/content/cell_images/upload/myphoto.png')
except:
  pass
```

![anvil-deploy](https://i.imgur.com/AcDJSqX.png)
