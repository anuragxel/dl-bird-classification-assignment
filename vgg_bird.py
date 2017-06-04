import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
import torch.optim as optim
from lr_scheduler import ReduceLROnPlateau

opt = {
    "num_classes" : 200,
    "lr" : 0.001, # As suggested in Part-based R-CNNs for Fine-grained Category Detection from RCNN paper
}

class bird_net(nn.Module):
    def __init__(self):
        super(bird_net, self).__init__()
        
        premodel = models.vgg16(pretrained=True) ## Ooh, VGG! :-*

        self.features = nn.Sequential(*list(premodel.features.children()))
        
        classifier_arr = list(premodel.classifier.children())[:-1] # Take all the FC layers but not the last one.
        classifier_arr.append(nn.Linear(4096, opt["num_classes"])) # Add a linear layer of number of classes
        
        init.kaiming_normal(classifier_arr[-1].weight) # Initialize with the He initialization for the last layer

        self.classifier = nn.Sequential(*classifier_arr)

    def forward(self,x):
        f = self.features(x)
       	return self.classifier(f)

birdie = bird_net()
print birdie
ignored_params = list(birdie.classifier.parameters())[-1]
ignored_params_id = list(list(map(id, birdie.classifier.parameters()))[-1]) # Take the last one
base_params = filter(lambda p: id(p) not in ignored_params_id,
                     birdie.parameters())

optimizer = optim.Adam([
                      { 'params' : base_params},
                      { 'params' : ignored_params, 'lr': opt['lr']}
                  ], lr = opt['lr']*0.1 )

lr_dampener = ReduceLROnPlateau(optimizer, 'min')

