import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    return train_data

def gpu():
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        print("running on GPU")
    else:
        device = 'cpu'
        print("running on CPU")
        
    return device

def parse():
    parser = argparse.ArgumentParser(description='Train a neural network with open of many options')
    parser.add_argument('--data_directory', default="flowers", help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--hidden_units',type=int, help='number of hidden units')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def def_data(data_dir):
    print("processing data into training data, test data, validation data and labels")
    train_dir, test_dir, valid_dir = data_dir 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir , transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    # Data batching
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    print("images in training dataset:", len(train_data))
    print("batches in training dataset:", len(trainloader))
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    return loaders

def getdata():
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir,test_dir,valid_dir]
    return def_data(data_dir)

#model specification (pre trained)
def specify_model(data): 
        
    if (args.arch is None):
        arch_type = 'densenet'       
    else:
        arch_type = args.arch
    if (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        model.name="densenet121"
        input_node=1024
        output_node=102
    elif (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        model.name="vgg19"
        input_node=25088
        output_node=500
    if (args.hidden_units is None):
        hidden_units = 512
    else:
        hidden_units = args.hidden_units
        
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    
    classifier= nn.Sequential(nn.Linear(input_node,hidden_units),
                          nn.ReLU(),
                          nn.Dropout(p=0.5),
                          nn.Linear(hidden_units, output_node),
                          nn.LogSoftmax(dim=1))
    model.classifier = classifier

    device=gpu()
    model.to(device);
    criterion = nn.NLLLoss()
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    return model

def validation(model, testloader, criterion): 
    device=gpu()
    model.to(device);
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)    
    
    
    for inputs, labels in testloader:
        inputs, labels=inputs.to(device), labels.to(device)
        logps =model.forward(inputs)
        batch_loss=criterion(logps, labels)
        test_loss+=batch_loss.item()
        ps=torch.exp(logps)
        top_p, top_class=ps.topk(1, dim=1)
        equals = top_class==labels.view(*top_class.shape)
        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
    return test_loss, int(accuracy)

def train(model,data):
    print("training model")
    
    print_every=10
    steps = 0
    
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    if (args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    device=gpu()

     
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    model.to(device)
    
    for e in range(epochs):        

        running_loss = 0 #calculate loss during training
        model.train()

        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward pass
            outputs = model.forward(images)#passing images
            loss = criterion(outputs, labels)#calculated loss 
            loss.backward()#calculates gradients
            optimizer.step()#update the weights

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()#turns dropout off

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Training Loss: {:.3f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.3f} | ".format(valid_loss/len(testloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
                running_loss = 0
                model.train()#turns dropout back on

    print("\nTraining process is completed.")
    return model

def save(model,train_data):
    print("saving model")
    if (args.save_dir is None):
        save_dir = 'check.pth'
    else:
        save_dir = args.save_dir

    model.class_to_idx = train_data.class_to_idx
        
    checkpoint = {'arch': model.name,
                  'model': model,
                  'classifier': model.classifier,
                  'features': model.features,
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    device =gpu()
    data =getdata()
    
    model = specify_model(data)
    model = train(model,data)
    train_dir = args.data_directory + '/train'
    train_data = train_transformer(train_dir)
    save(model,train_data)



def main():

    
    model = models.densenet121(pretrained=True)
    model.name = "densenet121"
    print("creating an image classifier")
    global args
    args = parse()
    device = gpu()
    
    
    train_dir = args.data_directory + '/train'
    train_data = train_transformer(train_dir)
    model=specify_model(train_data)
    model.to(device)
    create_model()
    print(model)
    return None

if __name__ == "__main__":
    main()
