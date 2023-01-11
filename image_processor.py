from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
import torch
import os

model_list = [model for model in dir(models) if not "_" in model and model.islower()]
T = transforms
num_classes = 0
batch_size = 64
normal = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
class_to_idx = None


train_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*normal)
])

test_transform = transforms.Compose([
#     transforms.CenterCrop(224),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
#     transforms.CenterCrop(224),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
      
        
def create_and_move_loader(dl, data_dir, transform, device):
    """
    Creates a dataset loader given a directory
    Returns None if loader_dir does not exists
    """
    folder = os.path.join(data_dir, dl)
    loader = None
    try:
        folder = datasets.ImageFolder(folder, transform=transform)
        if dl == "train":
            global num_classes
            num_classes = len(folder.classes)
            global class_to_idx
            class_to_idx = folder.class_to_idx
            loader = DataLoader(folder, batch_size=batch_size, shuffle=True, pin_memory=True)
        else:    
            loader = DataLoader(folder, batch_size=batch_size, pin_memory=True)
            
        loader = DeviceDataLoader(loader, device)
    except:
        print(f"{folder} not found")
    finally: 
        return loader
                                     
                                     
def create_dataloaders(data_dir, device=torch.device("cpu")):
    """
    Reads the input data directory and attempts to create
    train, valid and test pytorch dataloader instance and 
    moves them to the indicated device
    
    returns a dictionary containing train, valid, test ImageFolder instance
    """
    loader_dict = {"train":train_transform, "valid":valid_transform, "test":test_transform}
    for k, t in loader_dict.items():
        loader_dict[k] = create_and_move_loader(k, data_dir, transform=t, device=device)
    loader_dict["num_classes"] = num_classes  
    loader_dict["class_to_idx"] = class_to_idx
    return loader_dict


def process_image(image_path, device):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)
    
    # Resize image
    height, width = max(im.height, 256), max(im.width, 256)
    im.thumbnail(size=(height, width))
    
    new_width, new_height = (224, 224)
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    x_right = round(width - new_width) - left
    x_bottom = round(height - new_height) - top
    right = width - x_right
    bottom = height - x_bottom

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    normal = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*normal)
    ])

    # image to numpy
    np_image = np.array(norm_transform(im), dtype=np.float)
    
    # reorder color channel
    x =  torch.from_numpy(np_image).type(torch.FloatTensor)
    print(f"loading image to {device} ... ")
    xb = x.unsqueeze(0)
    xb = to_device(xb, device)
    return xb
    