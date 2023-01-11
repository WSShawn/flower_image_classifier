from model_utils import model_in_features

import argparse
import json
import os


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="mymodel.pth", help="path save model")
    parser.add_argument(
        "--arch", 
        type=str, 
        default="vgg13", 
        help=f"neural network architechture to use. available options are : {list(model_in_features.keys())}"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=15, help="number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    
    return parser.parse_args()

def parse_predict_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="number of top scoring classes to return")
    parser.add_argument(
        "--category_names", 
        type=str, 
        default="cat_to_name.json", 
        help="path to a json file containing a mapping of categorical name to index"
    )
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    
    return parser.parse_args()

                        
def print_result(epoch, result):
    print("Epoch [{}], train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch,
        result['train_loss'], 
        result['train_acc'], 
        result['val_loss'], 
        result['val_acc'])
    )


def get_class_to_name(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name