import argparse
import numpy as np

import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch.optim as optim
from simple_utils import ReduceLROnPlateau,EarlyStopping
from bird_dataset_generator import BirdClassificationGenerator


parser = argparse.ArgumentParser(description="Birds UCSD 200 2011 Classfication Assignment")
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('--arch','-a', metavar='ARCH', default='vgg',choices=['vgg','densenet'], help='model architecture: vgg|densenet (default: vgg)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size','-b',default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--print-freq', '-p',default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on test set')
args = parser.parse_args()
opt = {
    "num_classes" : 200,
}

class bird_net(nn.Module):
    def __init__(self, arch):
        super(bird_net, self).__init__()
        
        if arch == 'vgg':
            premodel = models.vgg16(pretrained=True) ## Ooh, VGG! :-*

            self.features = premodel.features
        
            classifier_arr = list(premodel.classifier.children())[:-1] # Take all the FC layers but not the last one.
            classifier_arr.append(nn.Linear(4096, opt["num_classes"])) # Add a linear layer of number of classes
        
            init.kaiming_normal(classifier_arr[-1].weight) # Initialize with the He initialization for the last layer

            self.classifier = nn.Sequential(*classifier_arr)
        elif arch == 'densenet':
            premodel = models.densenet121(pretrained=True)
            
            self.features = premodel.features
            self.classifier = nn.Linear(1024, opt["num_classes"])
            init.kaiming_normal(self.classifier.weight)
        else:
            raise RuntimeError("The architecture is not supported")
 
    def forward(self,x):
        f = self.features(x)
       	return self.classifier(f)


def train(network, optimizer):
    pass

def evaluate():
    pass

def accuracy():
    pass

def norm_confusion_matrix():
    pass

if __name__ == "__main__":
    birdie = bird_net(args.arch)
    print(birdie)
    ignored_params = list(map(id, birdie.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     birdie.parameters())

    optimizer = optim.SGD([
                      { 'params' : base_params},
                      { 'params' : birdie.classifier.parameters(), 'lr': args.lr}
                  ], lr=args.lr*0.1, momentum = 0.9 ) # Not trying really hard stuff here, just go with the paper mentioned above

    # Utility functions to reduce rate and stop training
    lr_dampener = ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=5) 
    train_stopper = EarlyStopping(optimizer, patience=10, mode='min')

    data = BirdClassificationGenerator(args.data, 0.2, args.batch_size)
    for values, bbs, labels in data.train_generator():
        print(values,labels, bbs)
        break

    for values, bbs in data.test_generator():
        print(values,bbs)
        break
    
    #for epoch in range(args.epochs):
    #    val_acc, val_loss = train(bird_net, optimizer)
    #    lr_dampner.step()
    #    if train_stopper.step(metric, epoch):
    #        # save model
    #        break
    #    pass

