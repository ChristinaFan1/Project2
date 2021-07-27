import argparse
import json
import PIL
import torch
import numpy as np
import os

from math import ceil
from PIL import Image

from  torchvision import models
def gpu():
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        print("running on GPU")
    else:
        device = 'cpu'
        print("running on CPU")
        
    return device

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--image_path', type=str, help='path of image to be predicted')
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='path to category to flower name mapping json')
    parser.add_argument('--checkpoint' , type=str, default='check.pth', help='path of saved trained model')
    parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
    parser.add_argument('--gpu',action="store_true",help='Use GPU + Cuda for calculations')
    args = parser.parse_args()
    return args

# Function load_checkpoint(checkpoint_path) loads saved trained model from checkpoint file
def load_checkpoint(checkpoint_path):
    """
    Loads deep learning model checkpoint.
    """
    
    if os.path.isfile(checkpoint_path):
       print("Loading checkpoint '{}'".format(checkpoint_path))
       checkpoint = torch.load(checkpoint_path)    
              
       model = getattr(models, checkpoint['arch'])(pretrained=True)
       device=gpu()
       model.to(device)
                   
       model.classifier = checkpoint['classifier']
       model.load_state_dict(checkpoint['state_dict'])
       model.class_to_idx=checkpoint['class_to_idx']
                
       return model
 


def processimage(image_path):
    image =Image.open(image_path)

    # Current dimensions
    width, height = image.size
    # resize the images and keep the aspect ratio
    image = image.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    width, height = image.size

    # create 224x224 image 
    center = width/2, height/2
    left=center[0]-(224/2)
    top=center[1]-(224/2)
    right=center[0]+(224/2)
    bottom =center[1]+(224/2)
    
    image = image.crop((left, top, right, bottom))

    # imshow() reqires binary(0,1) so divided by 255
    np_image = np.array(image)/255 
    
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    np_image=(np_image-mean)/std
    
    np_image=np_image.transpose((2,1,0))
       
    return torch.from_numpy(np_image)


def predict(image_tensor, model, device, cat_to_name, topk, image_path):
    image_torch = torch.from_numpy(np.expand_dims(image_tensor,axis=0)).type(torch.FloatTensor)                    
    device = gpu()    
    if (args.gpu):
        image_torch = image_torch.to(device)
        model.to(device)    
    else:
        image_torch = image_torch.to(device)
        model.to(device)

    
    model.to(device);
    model.eval();

    
    # probabilities-log softmax is on a log scale
    log_pr = model.forward(image_torch)
    #linear scale
    linear_pr = torch.exp(log_pr)
    #Top 5 predictions and labels
    top_pr, top_labels = linear_pr.topk(args.topk) 
    top_pr = np.array(top_pr.detach())[0]
    top_labels = np.array(top_labels.detach())[0]     

    idx_to_class={v:k for k,v in model.class_to_idx.items()}
    top_labels = [idx_to_class[label] for label in top_labels]
    top_fl = [cat_to_name[label] for label in top_labels]
    
    print(top_pr)
    print(top_fl)

    return top_pr, top_fl, top_labels
         
        
    

def printprobability(probs, flowers):
    #Convert two lists into a dictionary

    for i, j in enumerate(zip(probs, flowers)):
        print ("TopClass {}:".format(i+1),
               "Flower: {}, probability: {}%".format(j[0], ceil(j[1]*100)))
           

# Main Function
def main():
    
    global args 
   #Get Keyword Args for Prediction
    args =arg_parser()
    gpu()


    with open(args.cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint(args.checkpoint)
    
    image_tensor = processimage(args.image_path)
    
    device = gpu()
    image_path = args.image_path
    
    prediction = predict(image_tensor, model, device, cat_to_name, args.topk,image_path)
    top_pr, top_fl,top_labels = predict(image_tensor,model, device, cat_to_name, args.topk,image_path)
        
    printprobability(top_fl, top_pr)
    
    return prediction



if __name__ == '__main__': 
    main()