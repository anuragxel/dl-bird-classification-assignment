import numpy as np
#import cv2
import os
import csv
import random


def to_categorical(y,num_classes):
    y = np.array(y, dtype='int').ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

class BirdClassificationGenerator(object):
    def __init__(self, dataset_path, validation_ratio=0.3, batch_size=16):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_classes = 200

        self.train_list = []
        self.test_list = []
        
        self.bb_bird_dict = {}
        self.images_dict = {}
        
        self.train_labels_dict = {}
        
        with open(os.path.join(dataset_path, 'images_train.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                self.train_list.append(int(row[0]))
                self.images_dict[int(row[0])] = row[1]
                

        with open(os.path.join(dataset_path, 'bounding_boxes_train.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                self.bb_bird_dict[int(row[0])] = [ float(x) for x in row[1:4] ]
         
        with open(os.path.join(dataset_path, 'image_class_labels_train.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                self.train_labels_dict[int(row[0])] = int(row[1])
        
        with open(os.path.join(dataset_path, 'images_test.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                self.test_list.append(int(row[0]))
                self.images_dict[int(row[0])] = row[1]
                       
        with open(os.path.join(dataset_path, 'bounding_boxes_test.txt')) as f:
            spamreader = csv.reader(f, delimiter=' ')
            for row in spamreader:
                self.bb_bird_dict[int(row[0])] = [ float(x) for x in row[1:5] ]
        
        
        size_val_list = int(validation_ratio*len(self.train_list))
        self.val_list = random.sample(self.train_list, size_val_list)
        self.train_list = filter(lambda x: x not in self.val_list, self.train_list)
       
        # Coz Gradient Descent
        random.shuffle(self.val_list)
        random.shuffle(self.train_list)
        random.shuffle(self.test_list)

    def _generate(self,idx_list,set_type,batch_size):
        loop = 0
        epoch_iter = 0
        max_size = len(idx_list)
        while True:
            if set_type == "test" and epoch_iter == 1: 
                break
            if loop + batch_size < max_size:
                gen_list = idx_list[loop:loop+batch_size]
            else:
                last_iter = loop + batch_size - max_size
                gen_list = idx_list[loop:max_size] + gen_list[0:last_iter]
                loop = 0
                epoch_iter += 1
            loop += batch_size
            assert(len(gen_list) == batch_size)
            if set_type == 'train':
                yield ([ os.path.join(self.dataset_path, self.images_dict[x]) for x in gen_list ], 
                          np.array([ self.bb_bird_dict[x] for x in gen_list ]), 
                          to_categorical([ self.train_labels_dict[x] - 1 for x in gen_list ], self.num_classes)) 
            else:
                yield ([ os.path.join(self.dataset_path, self.images_dict[x]) for x in gen_list ], 
                           np.array([ self.bb_bird_dict[x] for x in gen_list ]))

    def train_generator(self):
        return self._generate(self.train_list, 'train', self.batch_size)
 
    def val_generator(self): 
        return self._generate(self.val_list, 'train', self.batch_size)

    def test_generator(self):
        return self._generate(self.test_list, 'test', self.batch_size)

if __name__ == "__main__":
    obj = BirdClassificationGenerator('/Neutron9/anurag/CUB_200_2011/',0.2,8)
    i = 1
    for paths, bbs, labels in obj.train_generator(): 
        print(paths,labels, bbs)
        break
    for values, bbs in obj.test_generator():
        print(paths,bbs)
