from torch import nn
import torch
import sys
import os

from utils import get_class_to_name, parse_predict_args
from image_processor import process_image
from model_utils import build_model

F = nn.functional
image_path = None
checkpoint_path = None
json_file = 'cat_to_name.json'
class_to_name = get_class_to_name(json_file)
images = ["jpeg", "png", "jpg"]


def get_image_checkpoint(paths, images=images):
    for path in paths:
        if path.split(".")[1] in images:
            global image_path
            image_path = path
        else:
            global checkpoint_path
            checkpoint_path = path

    if not image_path or not checkpoint_path:
        raise Exception(f"""
        First and second arguments must be paths to image and saved model. 
        Got {paths[0]} and {paths[1]}""")
    
        
def rebuild_model(path, device, train=False):
    "Loads model weights from checkpoint"
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(path, map_location=map_location)
    model = build_model(
        arch = checkpoint['model_name'],
        num_classes = checkpoint["num_classes"],
        hidden_units = checkpoint["hidden_units"],
        class_to_idx = checkpoint["class_to_idx"]
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print()
    print(f"Model parent name : {checkpoint['model_name']}")
    print(f"Number of classes trained on : {checkpoint['num_classes']}")
    print(f"Number of hidden units : {checkpoint['hidden_units']}")
    if not train:
        model.eval()
    return model.to(device)

def predict(image_path, model, k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    ''' 
    # load and process image
    print(f"reading and processing image from {image_path} ... ")
    xb = process_image(image_path, device)
    print("making predictions ... ")
    
    return model(xb).topk(k)

def print_prediction(prediction, cat_to_name, idx, classes):
    probs, klass = prediction
    
    # convert probs to probality distribution
    probs = torch.exp(F.log_softmax(probs, dim=-1)).tolist()[0]
    
    # get index of highest probality
    highest_prob = probs.index(max(probs))
    
    # convert to list
    # use model.class_to_idx[c]
    name = [cat_to_name[classes[idx.index(c)]] for c in klass.tolist()[0]]
    
    # sort name by probability
    name_probs = sorted(zip(name, probs), key = lambda x : x[1], reverse=True)
    print(f"Top {len(probs)} Model Predictions: ")
    for item in name_probs:
        print(f"{item[0]: <20} --- probability : {item[1]:.4f}")
     

def main():
    # get_inpur_args
    input_args = parse_predict_args()
    
    # load into device
    if input_args.gpu:
        print(f"Switching to GPU")
        if  torch.cuda.is_available():
            print(f"--gpu flag is used with no available gpu, reverting to cpu for predictions")
        else:
            device = torch.device("cuda")
    
    else:
        print(f"Using cpu for prediction")
        device = torch.device("cpu")
        print()       
    
    # load model
    print(f"Rebuilding model from checkpoint path {checkpoint_path}.... ")
    model = rebuild_model(path=checkpoint_path, device=device, train=False)
    print()
    
    # read cat_to_name.json file
    print(f"Getting categorical names from {input_args.category_names} ...")
    cat_to_name = get_class_to_name(input_args.category_names)
    idx = list(model.class_to_idx.values())
    classes = list(model.class_to_idx.keys())
    print()
    
    # make predictions
    prediction = predict(image_path, model, k=input_args.top_k, device=device)
    
    print_prediction(prediction, cat_to_name, idx, classes)
     
        
    
    
if __name__ == "__main__" :
    # remove input and checkpoint from args
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        main()        

    get_image_checkpoint(sys.argv[1:3], images=images)
    
    # remove first two arguments (image path and checkpoint path)
    sys.argv.pop(1)
    sys.argv.pop(1)
    
    main()
        
