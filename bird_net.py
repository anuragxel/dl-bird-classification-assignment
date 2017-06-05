import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable

from simple_utils import AverageMeter, ReduceLROnPlateau, EarlyStopping
from bird_dataset_generator import BirdClassificationGenerator


parser = argparse.ArgumentParser(description="Birds UCSD 200 2011 Classfication Assignment")
parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('--arch','-a', metavar='ARCH', default='vgg',choices=['vgg','densenet'], help='model architecture: vgg|densenet (default: vgg)')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size','-b',default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--print-freq', '-p',default=10, type=int, metavar='N', help='print frequency (default: 10)')
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
 
    def forward(self, x):
        f = self.features(x)
       	return self.classifier(f)

def accuracy(target_var, pred_var):
    _, pred = torch.max(pred_var, 1)
    _, target = torch.max(target_var,1)
    total = labels.size(0)
    correct = (pred == target).sum()
    return correct, total, ( 100 * (float(correct)/ float(total)))
    
def train(gen, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    
    net.train()
    for i, minibatch in enumerate(gen):
        idxes, images, labels = minibatch
        
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        input_var = Variable(images)
        target_var = Variable(labels)
   
        pred_var = net(input_var)

        # Calculate loss and accuracy
        loss = criterion(target_var, pred_var)
        correct, total, acc = accuracy(target_var,pred_var)
        
        # Update running averages
        losses.update(loss.data[0], images.size(0))
        accs.update(acc)

        # Perform Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print stuff
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, loss=losses, acc=acc))
            
def validate(gen, net, criterion):
    losses = AverageMeter()
    corrects = 0
    totals = 0
    
    net.eval()
    for i, minibatch in enumerate(gen):
        idxes, images, labels = minibatch
        
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        input_var = Variable(images)
        target_var = Variable(labels)
        
        pred_var = net(input_var)
        
        # Calculate loss and accuracy
        loss = criterion(target_var, pred_var)
        correct, total, acc = accuracy(target_var,pred_var)
        
        # Update stuff 
        losses.update(loss.data[0], images.size(0))
        corrects += correct
        totals += total
            
    acc = float(corrects)/float(totals)    
    return losses.avg, acc


def test(gen, net, criterion):
    net.eval()
    predictions = {}
    
    net.eval()
    for i, minibatch in enumerate(gen):
        
        idxes, images = minibatch
        images = torch.from_numpy(images)
        input_var = Variable(images)
        
        # Get predictions
        pred_var = net(input_var)
        _, pred = torch.max(pred_var, 1)
        
        # Store predictions
        for i, idx in enumerate(idxes):
           predictions[idx] = pred[i]
    return predictions


if __name__ == "__main__":
    birdie = bird_net(args.arch)
    print(birdie)
    ignored_params = list(map(id, birdie.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     birdie.parameters())

    birdie.features = torch.nn.DataParallel(birdie.features)
    birdie.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([
                      { 'params' : base_params},
                      { 'params' : birdie.classifier.parameters(), 'lr': args.lr}
                  ], lr=args.lr*0.1, momentum = 0.9 ) # Not trying really hard stuff here, just going with the paper mentioned above

    # Utility functions to reduce rate and stop training
    lr_dampener = ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=5) 
    train_stopper = EarlyStopping(optimizer, patience=10, mode='min')

    datagen = BirdClassificationGenerator(args.data, 0.2, args.batch_size)
    best_acc = 0
    for epoch in range(args.epochs):
        train(datagen.train_generator(), birdie, criterion, optimizer, epoch)
        val_loss, val_acc = validate(datagen.val_generator(), birdie, criterion)
        lr_dampner.step(val_loss, epoch)
        if val_acc > best_acc:
            print("Accuracy improved to %s from %s" % val_acc, best_acc)
        best_acc = max(val_acc, best_acc)
        if train_stopper.step(val_loss, epoch):
            save({
            	'epoch': epoch + 1,
            	'arch': args.arch,
           	'state_dict': birdie.state_dict(),
            	'best_acc': best_acc,
            	'optimizer' : optimizer.state_dict(),
        	}, 'best_checkpoint.pth.tar')
            break
    predictions = test(datagen.test_generator(), birdie, criterion)       

      
