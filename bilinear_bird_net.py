import os
import csv
import argparse
import numpy as np

try:
    import ipdb as pdb
except:
    import pdb

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
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size','-b',default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--fine-tune','-f',default=0, type=int, metavar='N', help='If you have already trained the final layer, then fine-tune the network by setting it to 1')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--print-freq', '-p',default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval', default='', metavar='DIR', help='evalaute model at path')
args = parser.parse_args()

class bilinear_cnn_dd(nn.Module):
    def __init__(self, num_classes):
        super(bilinear_cnn_dd, self).__init__()

        premodel = models.vgg16(pretrained=True) ## Ooh, VGG! :-*
        
        ## The end feature should be conv5_3, so we need to remove the maxpool
        features_arr = list(premodel.features.children())[:-1] # remove the maxpool
        self.features = nn.Sequential(*features_arr)        

        ## The classifier is supposed to be a softmax, so add a linear layer of "size phi_I*num_classes"
        classifier_arr = []
        classifier_arr.append(nn.Linear(512*512, 200))

        init.kaiming_normal(classifier_arr[0].weight)
        
        self.classifier = nn.Sequential(*classifier_arr)

    def forward(self, inp):
        f = self.features(inp) # batch, 512, 7, 7

        x_ = f.view(args.batch_size, f.size(1), -1) # batch, filters, h*w
        x__ = torch.transpose(x_, 1, 2) # batch, h*w, filters
        
        # Perform the bilinear operation and flatten the array
        phi_I = torch.bmm(x_,x__) # batch, 512, 512
        phi_I = phi_I.view(args.batch_size,-1) # batch, 512*512
        
        # Divide by num_dims
        phi_I = torch.div(phi_I, 784.0)

        # Peform the signed sqrt
        sign_sqrt = torch.mul(torch.sqrt(torch.abs(phi_I)) + 1e-12, torch.sign(phi_I))
        
        # Perform the l2 normalization
        phi_l2_normed = sign_sqrt / (sign_sqrt.norm(p=2, dim=1).expand_as(sign_sqrt) + 1e-6)
       	return self.classifier(phi_l2_normed)

def accuracy(pred_var, target_var):
    _, pred = torch.max(pred_var, 1)
    total = target_var.size(0)
    correct = (pred == target_var).sum()
    correct = correct.data.cpu().numpy()[0]
    return correct, total, ( 100 * (float(correct)/ float(total)))
    
def train(gen, net, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    
    net.train()
    for i, minibatch in enumerate(gen):
        idxes, images, labels = minibatch
        
        images = torch.from_numpy(images).cuda()
        labels = torch.from_numpy(labels).cuda()

        input_var = Variable(images)
        target_var = Variable(labels)
        
        pred_var = net(input_var)

        # Calculate loss and accuracy
        loss = criterion(pred_var, target_var)
        correct, total, acc = accuracy(pred_var, target_var)
        
        # Update running averages
        losses.update(loss.data[0], images.size(0))
        accs.update(acc)

        # Perform Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print stuff
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch + 1, i, loss=losses, acc=accs))
              #    ' Add acc'.format(epoch, i, loss=losses))
            
def validate(gen, net, criterion):
    losses = AverageMeter()
    corrects = 0
    totals = 0
    
    net.eval()
    for i, minibatch in enumerate(gen):
        idxes, images, labels = minibatch
        
        images = torch.from_numpy(images).cuda()
        labels = torch.from_numpy(labels).cuda()

        input_var = Variable(images, volatile=True)
        target_var = Variable(labels, volatile=True)
        
        pred_var = net(input_var)
        
        # Calculate loss and accuracy
        loss = criterion(pred_var, target_var)
        correct, total, acc = accuracy(pred_var, target_var)
        
        # Update stuff 
        losses.update(loss.data[0], images.size(0))
        corrects += correct
        totals += total

    acc = float(corrects)/float(totals)
    return losses.avg, acc


def predict(gen, net, criterion):
    net.eval()
    predictions = {}
    
    net.eval()
    for i, minibatch in enumerate(gen):
        
        idxes, images = minibatch
        images = torch.from_numpy(images).cuda()
        input_var = Variable(images, volatile=True)
        
        # Get predictions
        pred_var = net(input_var)
        _, pred = torch.max(pred_var, 1)
        
        # Store predictions
        for i, idx in enumerate(idxes):
           predictions[idx] = pred[i].data.cpu().numpy()[0]
    return predictions

if __name__ == "__main__":
    birdie = bilinear_cnn_dd(200) 
    print(birdie)
    if args.fine_tune == 0: # If not fine tuning, freeze the layers
        for param in birdie.features.parameters():
            param.requires_grad = False
    
    ignored_params = list(map(id, birdie.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     birdie.parameters())

    birdie.features = torch.nn.DataParallel(birdie.features)
    birdie.cuda()
    
    optimizer = optim.SGD([
                        { 'params' : birdie.classifier.parameters(), 'lr': args.lr }
                    ], lr=args.lr, momentum = 0.9) # Train only the classifier, freeze the other weights 
            
    criterion = nn.CrossEntropyLoss().cuda()
    datagen = BirdClassificationGenerator(args.data, 0.2, args.batch_size)
    # Utility functions to reduce rate and stop training
    if args.eval != '':
        checkpoint = torch.load(args.eval)
        args.epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        birdie.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        predictions = predict(datagen.test_generator(), birdie, criterion)
    else:
        if args.fine_tune == 1:
            if not os.path.exists('first_step_best.pth.tar'):
               raise Exception("First Step is not complete! Can't fine-tune.")
            checkpoint = torch.load('first_step_best.pth.tar')
            best_acc = checkpoint['best_acc']
            birdie.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param in birdie.features.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            print("Fine Tuning network")
            train_stopper = EarlyStopping(optimizer, patience=1, mode='min')             
            for epoch in range(args.epochs):
                train(datagen.train_generator(), birdie, criterion, optimizer, epoch)
                val_loss, val_acc = validate(datagen.val_generator(), birdie, criterion)
                print("Epoch: " + str(epoch + 1) + " Val Acc: " + str(val_acc) + " Val Loss: " + str(val_loss))
                if val_acc > best_acc:
                    print("Accuracy improved to " + str(val_acc) + " from " + str(best_acc))
                    torch.save({
            	        'epoch': epoch + 1,
           	        'state_dict': birdie.state_dict(),
            	        'best_acc': val_acc,
            	        'optimizer' : optimizer.state_dict(),
        	     }, 'best_checkpoint_bilinear_bird.pth.tar') 
                best_acc = max(val_acc, best_acc)
                if epoch % 5 == 0 and epoch != 0:
                    torch.save({
            	        'epoch': epoch + 1,
           	        'state_dict': birdie.state_dict(),
            	        'best_acc': best_acc,
            	        'optimizer' : optimizer.state_dict(),
        	    }, 'epoch_'+ str(epoch) + '_bilinear_bird.pth.tar')
                if train_stopper.step(val_loss, epoch):
                    break
        else:
            train_stopper = EarlyStopping(optimizer, patience=1, mode='min')
            best_acc = 0
            for epoch in range(args.epochs):
                train(datagen.train_generator(), birdie, criterion, optimizer, epoch)
                val_loss, val_acc = validate(datagen.val_generator(), birdie, criterion)
                print("Epoch: " + str(epoch + 1) + " Val Acc: " + str(val_acc) + " Val Loss: " + str(val_loss))
                if val_acc > best_acc:
                    print("Accuracy improved to " + str(val_acc) + " from " + str(best_acc))
                    torch.save({
            	    'epoch': epoch + 1,
           	    'state_dict': birdie.state_dict(),
            	    'best_acc': val_acc,
            	    'optimizer' : optimizer.state_dict(),
        	    }, 'best_checkpoint_bilinear_bird.pth.tar') 
                best_acc = max(val_acc, best_acc)
                if epoch % 5 == 0 and epoch != 0:
                    torch.save({
            	    'epoch': epoch + 1,
           	    'state_dict': birdie.state_dict(),
            	    'best_acc': best_acc,
            	    'optimizer' : optimizer.state_dict(),
        	    }, 'epoch_'+ str(epoch) + '_bilinear_bird.pth.tar')
                if train_stopper.step(val_loss, epoch):
                    break
    predictions = predict(datagen.test_generator(), birdie, criterion)
    with open('predictions.txt','w') as f:
        spamwriter = csv.writer(f, delimiter=',')
        spamwriter.writerow(['Id','Category'])
        for key, value in predictions.iteritems():
            spamwriter.writerow([key,value + 1])
