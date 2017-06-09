import numpy as np
import cv2
import os
import csv
import random

def get_image_region(im_path, bb):
    im = cv2.imread(im_path)
    bb = [ int(x) for x in bb ]
    x, y, w, h = bb
    im = im[y:y+h,x:x+w]
    im = cv2.resize(im, (224,224))
    im = im.astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68 
    im = np.divide(im, 255.0)
    im = np.transpose(im, (2,0,1))
    return im


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
                self.bb_bird_dict[int(row[0])] = [ float(x) for x in row[1:5] ]
         
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
        self._shuffle()

    def _shuffle(self): 
        # Coz Gradient Descent
        random.shuffle(self.val_list)
        random.shuffle(self.train_list)
    
    def _generate(self,idx_list,set_type,batch_size):
        loop = 0
        epoch_iter = 0
        max_size = len(idx_list)
        while True:
            if epoch_iter == 1: 
                break
            if loop + batch_size < max_size:
                gen_list = idx_list[loop:loop+batch_size]
                loop += batch_size
            else:
                last_iter = loop + batch_size - max_size
                gen_list = idx_list[loop:max_size] + gen_list[0:last_iter]
                loop = 0
                self._shuffle()
                epoch_iter += 1
            assert(len(gen_list) == batch_size)
            if set_type == 'train':
                yield ( gen_list, np.array([ get_image_region(os.path.join(self.dataset_path, 'images', self.images_dict[x]), self.bb_bird_dict[x]) for x in gen_list ]), 
                          np.array([ self.train_labels_dict[x] - 1 for x in gen_list ], dtype=np.int64).ravel())
                          #to_categorical([ self.train_labels_dict[x] - 1 for x in gen_list ], self.num_classes))
            else:
                yield ( gen_list, np.array([ get_image_region(os.path.join(self.dataset_path, 'images', self.images_dict[x]), self.bb_bird_dict[x]) for x in gen_list ]))

    def train_generator(self):
        return self._generate(self.train_list, 'train', self.batch_size)
 
    def val_generator(self): 
        return self._generate(self.val_list, 'train', self.batch_size)

    def test_generator(self):
        return self._generate(self.test_list, 'test', self.batch_size)

if __name__ == "__main__":
    obj = BirdClassificationGenerator('/Neutron9/anurag/CUB_200_2011/',0.2,8)
    for idxes, images, labels in obj.train_generator(): 
        print(idxes, images.shape,labels.shape)
        break
    for idxes, images, labels in obj.val_generator():
        print(idxes, images.shape, labels.shape)
        break
    for idxes, images in obj.test_generator():
        print(idxes,images.shape)
        break
        
