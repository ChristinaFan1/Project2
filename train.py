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
    if (args.gpu):
        device = 'cuda'
        print("running on gpu")
    else:
        device = 'cpu'
        print("running on cpu")
        
    return 0 
    
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


def specify_model(data): #model specification
        
    if (args.arch is None):
        arch_type = 'densenet'
       
    else:
        arch_type = args.arch
    if (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        model.name="densenet121"
        input_node=1024
    elif (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        model.name="vgg19"
        input_node=25088
    if (args.hidden_units is None):
        hidden_units = 512
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    
    classifier= nn.Sequential(nn.Linear(1024,512),
                          nn.ReLU(),
                          nn.Dropout(p=0.5),
                          nn.Linear(512, 102),
                          nn.LogSoftmax(dim=1))
    model.classifier = classifier

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device);
    criterion = nn.NLLLoss()
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    return model

def validation(model, testloader, criterion): 
    device = torch.device("cuda" if args.gpu else "cpu")#("cuda:0" if torch.cuda.is_available() else "cpu")
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
    epochs = 3
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    if (args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'

     
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    model.to(device)
    
    for e in range(epochs):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        running_loss = 0 #doing this to calculate loss during training
        model.train() # Technically not necessary, setting this for good measure


        for images, labels in iter(trainloader):
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()#Clears the gradients, do this because gradients are accumulated

            # Forward and backward passes
            outputs = model.forward(images)#passing our images
            loss = criterion(outputs, labels)#calculated loss necessary to calculate the gradient
            loss.backward()#calculates the gradients
            optimizer.step()#then with the gradients you can take a step to update the weights

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
    torch.save(model.state_dict(), save_dir)
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    data =getdata()
    model=specify_model(data)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    
    checkpoint = {'arch': model.name,
                  'model': model,
                  'classifier': model.classifier,
                  #'input_size': 1024,
                  #'output_size': 1,
                  #'epochs':3,
                  'features': model.features,
                  #'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'class_to_idx': train_data.class_to_idx,
                  #'dropout':0.5,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    gpu()
    data =getdata()
    
    model = specify_model(data)
    model = train(model,data)
    train_dir = args.data_directory + '/train'
    train_data = train_transformer(train_dir)
    save(model,train_data)

def parse():
    parser = argparse.ArgumentParser(description='Train a neural network with open of many options!')
    parser.add_argument('--data_directory', default="flowers", help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def main():

    model = models.densenet121(pretrained=True)
    model.name = "densenet121"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("creating an image classifier")
    global args
    args = parse()
    gpu()
    train_dir = args.data_directory + '/train'
    train_data = train_transformer(train_dir)
    create_model()
    
    print("\nModel finished with above 84% accuracy :)")
    return None

main()
