#!/usr/bin/env python
# coding: utf-8

# # Face detection and recognition training pipeline
# 
# The following example illustrates how to fine-tune an InceptionResnetV1 model on your own dataset. This will mostly follow standard pytorch training patterns.

# In[17]:


get_ipython().run_line_magic('cd', '../')
get_ipython().system('ls')


# In[18]:


from custom_dataset_from_csv import CustomDatasetFromCsvLocation
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


# #### Define run parameters
# 
# The dataset should follow the VGGFace2/ImageNet-style directory layout. Modify `data_dir` to the location of the dataset on wish to finetune on.

# In[23]:


data_dir = '../data/test_images'

batch_size = 32
epochs = 8
workers = 0 #  if os.name == 'nt' else 8


# #### Determine if an nvidia GPU is available

# In[24]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# #### Define MTCNN module
# 
# See `help(MTCNN)` for more details.

# In[25]:


"""
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
"""


# #### Perfom MTCNN facial detection
# Iterate through the DataLoader object and obtain cropped faces.
# 

# In[26]:


"""
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]
        
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print(x)
    print(y)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
# Remove mtcnn to reduce GPU memory usage
del mtcnn
"""


# In[27]:


# len(dataset.class_to_idx)


# #### Define Inception Resnet V1 module
# 
# See `help(InceptionResnetV1)` for more details.

# In[28]:


resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=2
).to(device)


# #### Define optimizer, scheduler, dataset, and dataloader

# In[29]:


"""
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)
"""


# In[30]:


optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])


# In[ ]:





# In[31]:



csv_path = './labels_2021-06-07-classification_imahira.csv'
data_dir = '../../dataset/puri_dataset/puri_face_Ariel_TOY_1024/'
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

phase = 'training'
train_dataset = CustomDatasetFromCsvLocation(
            csv_path,
            data_dir,
            phase,
            trans
        )
phase = 'validation'
val_dataset = CustomDatasetFromCsvLocation(
            csv_path,
            data_dir,
            phase,
            trans
        )
phase = 'test'
test_dataset = CustomDatasetFromCsvLocation(
            csv_path,
            data_dir,
            phase,
            trans
        )

train_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    num_workers=workers,
    batch_size=batch_size,
   shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    num_workers=workers,
    batch_size=batch_size,
   shuffle=True
)


# #### Define loss and evaluation functions

# In[32]:


loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}


# #### Train model

# In[34]:


writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()


# In[ ]:




