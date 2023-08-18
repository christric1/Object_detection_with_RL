import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

#------------------------------------------------------#
# name1 : Yolov7 label order
# name2 : Origin dataset label order
#------------------------------------------------------#
names1 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
names2 = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'couch', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
map_index_coco = [names1.index(i) for i in names2]

names3 = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
map_index_pascal = [names1.index(i) for i in names3]

#------------------------------------------------------#
#   Reference from : https://reurl.cc/Q4EEEZ
#------------------------------------------------------#
class OD_Dataset(Dataset): 
    '''
        root       : dataset path
        transforms : images transforms function
        mode       : train & valid
    '''
    def __init__(self, root, transforms=None, mode='train'):
        self.root = root 
        self.transforms = transforms
        self.images_dir = os.path.join(self.root, mode, "images")  
        self.annotations_dir = os.path.join(self.root, mode, "labels")  
       
        self.data_names = [os.path.splitext(f)[0] for f in os.listdir(self.images_dir) 
                           if os.path.isfile(os.path.join(self.images_dir, f))]
        
        # Check ignore.txt exist
        if "ignore.txt" not in os.listdir(os.path.join(self.root, mode)):
            ignore_list = []
            pbar = tqdm(self.data_names, desc="Create "+mode+" ignore.txt")
            for name in pbar:
                annotation_path = os.path.join(self.annotations_dir, name + ".txt")
                target = self.read_txt(annotation_path)

                # check whether is None
                if len(target) == 0:
                    ignore_list.append(name)
            
            docname = "./ignore.txt"
            path = os.path.join(self.root, mode, docname) 
            with open(path, 'w') as f:
                for line in ignore_list:
                    f.write(line + '\n')

        # delete useless 
        with open(os.path.join(self.root, mode, "ignore.txt"), 'r') as f:
            ignore_lines = [line.strip() for line in f.readlines()]
            self.data_names = [x for x in self.data_names if x not in ignore_lines]

    def __len__(self):
        '''
            return the length of dataset
        '''
        return len(self.data_names)

    def __getitem__(self, index):
        '''
            image_data : (3, height, width)
            target     : [label, xmin, ymin, xmax, ymax]
        '''
        # Read image
        image_path = os.path.join(self.images_dir, self.data_names[index] + ".jpg")
        image = Image.open(image_path).convert("RGB")  # PIL image

        # Read Annitation
        annotation_path = os.path.join(self.annotations_dir, self.data_names[index] + ".txt")
        target = self.read_txt(annotation_path)
        # target[:, 0] = np.array([map_index_pascal[i] for i in target[:, 0].astype(int)]).astype(float) # map

        # Transform to tensor
        target_tensor = torch.as_tensor(target)
        image_tensor  = F.to_tensor(image)

        if self.transforms is not None:  
            image, target = self.transforms(image, target)

        return image_tensor, target_tensor, image_path

    def read_txt(self, annotation_path):  # Read txt
        '''
            read from annotation txt,
            transforms the (X_center, Y_center, width, height) to (xmin, ymin, xmax, ymax)
        '''
        target = []
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip().split()
            label = float(line[0])

            x, y, w, h = map(float, line[1:])
            xmin = (x - w / 2)
            ymin = (y - h / 2)
            xmax = (x + w / 2)
            ymax = (y + h / 2)
            target.append((label, xmin, ymin, xmax, ymax))

        return np.array(target)