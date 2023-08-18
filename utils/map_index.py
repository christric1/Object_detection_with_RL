import os
import argparse
from tqdm import tqdm


#------------------------------------------------------#
# name1 : Yolov7 label order
# name2 & name3 : Origin dataset label order
#------------------------------------------------------#
names1 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
names2 = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'couch', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']
map_index_coco = [names1.index(i) for i in names2]


if __name__ == '__main__':
    '''
        Since the output label index of the model is different from the label index of the dataset, 
        conversion is required.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='coco')
    opt = parser.parse_args()

    root = opt.dataset_path 
    mode = ["train", "valid"]

    for m in mode:
        annotations_dir = os.path.join(root, m, "labels")
        for i in tqdm(os.listdir(annotations_dir), desc=m):
            txt_path = os.path.join(annotations_dir, i)

            with open(txt_path, 'r') as f:
                lines = f.readlines()
                new_lines = []
                for line in lines:
                    new_label = str(map_index_coco[int(line.split()[0])])
                    new_line = line.split()[1:]
                    new_line.insert(0, new_label)
                    new_lines.append(" ".join(new_line))

            with open(txt_path, 'w') as f:
                f.write('\n'.join(new_lines))