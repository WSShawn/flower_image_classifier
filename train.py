from tqdm import tqdm
from torch import nn

import torch
import sys
import os

from utils import print_result, parse_train_args
from image_processor import create_dataloaders
from workspace_utils import active_session
from model_utils import build_model

F = nn.functional
data_dir = "flowers"

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def validation_step(model, batch):
        images, labels = batch 
        out = model(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def evaluate(model, val_loader):
    with torch.no_grad():
        model.eval()
        outputs = [validation_step(model, batch) for batch in val_loader]
        return validation_epoch_end(outputs)

def test(model, test_loader):
    result = evaluate(model, test_loader)
    return {"test_loss" : result["val_loss"], "test_accuracy" : result["val_acc"]}

def single_train_step(model, batch):
    images, labels = batch 
    out = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    acc = accuracy(out, labels)          # Calculate accuracy
    return loss, acc

def fit_one_cycle(epochs, lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_accuracy = []
        for batch in tqdm(train_loader):
            # zero grad
            optimizer.zero_grad()
            
            loss, acc = single_train_step(model, batch)
            train_losses.append(loss)
            train_accuracy.append(acc)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accuracy).mean().item()
        print_result(epoch, result)
        history.append(result)
    return history
    
# TODO: Save the checkpoint
def save_model(path, model):
    path = path if len(path.split('.')) == 2 else os.path.join(path, model.arch_name + ".pth")
    torch.save({
        "model_state_dict" : model.state_dict(),
        "class_to_idx" : model.class_to_idx,
        "model_name" : model.arch_name,
        "num_classes" : model.num_classes,
        "hidden_units" : model.num_hidden_units,
    }, path)
    
def freeze_feature_parameters(model):
    for param in model.parameters():
        param.require_grad = False
    
    for param in getattr(model, "classifier", "fc").parameters():
        param.require_grad = True      

def main():
    history = []
    # get input args
    input_args = parse_train_args()
    
    if input_args.gpu:
        print(f"Switching to GPU")
        if not torch.cuda.is_available():
            print(f"--gpu flag is used with no available gpu, reverting to cpu for training")
    else:
        print(f"Using cpu for training")
            
    device = torch.device("cuda" if torch.cuda.is_available() and input_args.gpu else "cpu")
    
    # get loaders and num_classes
    print()
    print(f"preparing dataloaders from directory {data_dir}.... ")
    loader_dict = create_dataloaders(data_dir, device)
    train_dl = loader_dict['train']
    val_dl = loader_dict['valid']
    test_dl = loader_dict['test']
    num_classes = loader_dict["num_classes"]
    class_to_idx = loader_dict["class_to_idx"]
    print(f"found {num_classes} number of classes in train directory")
    print()
        
    # build model
    # if no hidden units is provided, use double the number of classes 
    # as hidden units
    hidden_units = input_args.hidden_units 
    print(f"building {input_args.arch} pretrained model with hidden_units {hidden_units} ")
    model = build_model(input_args.arch, num_classes, hidden_units, class_to_idx)
    model.to(device)
    freeze_feature_parameters(model)
    print()
    
    # training configurations
    learning_rate = input_args.learning_rate
    epochs = input_args.epochs
    optim_func = torch.optim.Adam
    grad_clip = 1
    weight_decay = 1e-6
    
    print(f"setting hyperparameters")
    print(f"number of epochs : {epochs}")
    print(f"learning_rate : {learning_rate}")
    print(f"optimizer function : {str(optim_func)}")
    print()
    
    print("Evaluate model on valid dataset before training")
    history = [evaluate(model, val_dl)]
    print(history)
    
    print("Beginning Training")
    with active_session():
        history += fit_one_cycle(epochs, learning_rate, model, train_dl, val_dl,
                             weight_decay=weight_decay,
                             grad_clip=grad_clip, 
                             opt_func=optim_func)
        

    print("Training Ended")
    print()
    print("Evaluating model on test set")
    print(test(model, test_dl))    
    print()
    print(f"Saving model to {input_args.save_dir}")  
    save_model(input_args.save_dir, model)
    print(f"model saved.")
    
    
if __name__ == "__main__":
    # data_dir is the first argument passed to train.py
    # remove data_dir from sys argv to avoid errors with argparser
    if os.path.isdir(sys.argv[1]):
        data_dir = sys.argv.pop(1)
        
    elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
        main()        
    else:
        raise Exception("First argument must be a parent directory to train, valid and test directory")
        
    main()