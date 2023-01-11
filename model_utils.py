from torchvision import models
from torch import nn

import torch

model_in_features= {
    "resnet34" : 512,  
    "vgg13" : 25088, 
    "densenet121" : 1024, 
    "alexnet" : 4096
}

F = nn.functional
classifier_output_layer = ["alexnet", "vgg13", "densenet121"]
fc_output_layer = ["resnet34"]

model_list = classifier_output_layer + fc_output_layer


class ModelNotFoundException(Exception):
    pass
                
def _build_output_layer(in_features, hidden_units, num_classes):
    n_h = int(hidden_units / 2) 
    classifier =  nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, n_h),
        nn.ReLU(),
        nn.Linear(n_h, num_classes),
    )    
    for param in classifier.parameters():
        param.require_grad = True
    
    return classifier

def customize_model_output(model, name, num_classes, hidden_units=None):
    """
    Replaces the last layer of a model 
    with a layer suitable for flower classification
    
    model : Pretrained model from torchvision
    name [string] : name of architecture of model
    num_classes [int] : number of classes in train dataset 
    """
    
    # freeze feature parameters    
    in_features = model_in_features[name]
    
    if name in classifier_output_layer:
        model.classifier = _build_output_layer(in_features, hidden_units, num_classes)

            
    elif name in fc_output_layer:
        model.fc = _build_output_layer(in_features, hidden_units, num_classes)
        
    else:
        raise ModelNotFoundError(f"""
        model with {name} is not one of the current model options: {list(model_in_features.keys())}.
        \nThis could also be caused by wrong spelling of model name.
        """)
    return model

def get_pretrained_model(arch):
    """
    Initialize  and returns a pretrained model
    
    arch : name of pretrained model
    """
    if not arch in model_list:
        raise ModelNotFoundException("the model {} could not be found in torchvision.model".format(arch))
    arch = getattr(models, arch)
    model = arch(pretrained=True)
    return model
        

def build_model(arch, num_classes, hidden_units, class_to_idx):
    pretrained_model = get_pretrained_model(arch)
    model = customize_model_output(pretrained_model, arch, num_classes, hidden_units)
    model.arch_name = arch
    model.num_classes = num_classes
    model.num_hidden_units = hidden_units
    model.class_to_idx = class_to_idx
    return model
   
