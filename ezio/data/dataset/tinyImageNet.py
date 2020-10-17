from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook
import zipfile
import requests
from io import StringIO,BytesIO

class TinyImageNet(Dataset):
    def __init__(self, classes, url):
        self.data = []
        self.target = []
        self.classes = classes
        self.url = url
        
        wnids = open(f"{url}/wnids.txt", "r")
        
        for wclass in notebook.tqdm(wnids, desc='Loading Train Folder', total = 200):
          wclass = wclass.strip()
          for i in os.listdir(url+'/train/'+wclass+'/images/'):
            img = Image.open(url+"/train/"+wclass+"/images/"+i)
            npimg = np.asarray(img)
            if(len(npimg.shape) == 2):
               npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
            self.data.append(npimg)  
            self.target.append(self.classes.index(wclass))

        val_file = open(f"{url}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(val_file,desc='Loading Test Folder',total =10000 ):
          split_img, split_class = i.strip().split("\t")[:2]
          img = Image.open(f"{url}/val/images/{split_img}")
          npimg = np.asarray(img)
          if(len(npimg.shape) ==2):
                    
                npimg = np.repeat(npimg[:, :, np.newaxis], 3, axis=2)
          self.data.append(npimg)  
          self.target.append(self.classes.index(split_class))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        img = data     
        return data,target


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)



def TinyImageNetDataSet(train_split = 70,test_transforms = None,train_transforms = None):
  url  = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  download_images(url)
  classes = []
  f = open("tiny-imagenet-200/wnids.txt", "r")
  for line in f:
    classes.append(line.strip())
  dataset = TinyImageNet(classes, url="tiny-imagenet-200")
  train_len = len(dataset)*train_split//100
  test_len = len(dataset) - train_len 
  train_set, val_set = random_split(dataset, [train_len, test_len])
  train_set = DatasetFromSubset(train_set, transform=train_transforms)
  test_set = DatasetFromSubset(val_set, transform=test_transforms)
  return train_set, test_set, classes

def download_images(url):
    if (os.path.isdir("tiny-imagenet-200")):
        print ('Images already downloaded...')
        return
    r = requests.get(url, stream=True)
    print ('Downloading TinyImageNet Data' )
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
      zip_ref.extract(member = file)
    zip_ref.close()
