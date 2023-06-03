import json
import os
import torch
import random
import numpy as np
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
from pprint import PrettyPrinter
from tqdm import tqdm
from pycocotools.coco import COCO
from itertools import chain
import datetime
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp = PrettyPrinter()
# Label map
coco_labels = ('person', 'car', 'stop sign','traffic light', 'motorcycle','boat', 'truck','bus','airplane','suitcase')
# voc_labels = ('Camera', 'Mobilephone', 'Smartphone')
label_map = {k: v + 1 for v, k in enumerate(coco_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
# distinct_colors = ['#AA4A44','#e6194b', '#3cb44b', '#ffe119',]
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    #print("Annotation Path"+annotation_path)
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        #difficult = int(object.find('difficult').text == '1')
        difficult=0
        label = object.find('name').text.strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties':difficulties}


def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    #voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    train_images = list()
    train_objects = list()
    n_objects = 0
    # test data
    test_images = list()
    test_objects = list()
    n_objects_test = 0
    # val data
    val_images = list()
    val_objects = list()
    n_objects_val = 0

    # Training data
    for path in [voc07_path]:

        data_folders = os.listdir(path+'/Annotations')
        print(path, data_folders)
        for data_folder in data_folders:
            # ignore the test images folder
            if 'test' not in data_folder.lower():

                if 'smart' in data_folder.lower():
                    print('GOT SMARTPHONE')
                    #obj_type = 'Smartphone'
                elif 'mobile' in data_folder.lower():
                    print('GOT MOBILEPHONE')
                    #obj_type = 'Mobilephone'
                elif 'camera' in data_folder.lower():
                    print('GOT CAMERA')
                    #obj_type = 'Camera'
                else:
                    continue
                print(data_folder)
            text_files = os.listdir(os.path.join(path+'/Annotations', data_folder))
            full_data=np.array(text_files)
            np.random.shuffle(full_data)
            split1 = int(0.7 * len(full_data))
            split2 = int(0.9 * len(full_data))
            train_arr = full_data[:split1]
            val_arr = full_data[split1:split2]
            test_arr = full_data[split2:]
            print("length of text files "+data_folder+":"+str(len(train_arr)))
                
            for file in train_arr:
                # parse text files
                objects = parse_annotation(os.path.join(path+'/Annotations', 
                                                data_folder, file.split('.')[0] + '.xml'))
                if len(objects) == 0:
                    continue
                n_objects += len(objects)
                train_objects.append(objects)
                train_images.append(os.path.join(path, 'JPEGImages', data_folder, file.split('.')[0] + '.jpg'))

            for file in val_arr:
                # parse text files
                objects = parse_annotation(os.path.join(path+'/Annotations', 
                                                data_folder, file.split('.')[0] + '.xml'))
                if len(objects) == 0:
                    continue
                n_objects_val += len(objects)
                val_objects.append(objects)
                val_images.append(os.path.join(path, 'JPEGImages', data_folder, file.split('.')[0] + '.jpg'))    

            for file in test_arr:
                # parse text files
                objects = parse_annotation(os.path.join(path+'/Annotations', 
                                                data_folder, file.split('.')[0] + '.xml'))
                if len(objects) == 0:
                    continue
                n_objects_test += len(objects)
                test_objects.append(objects)
                test_images.append(os.path.join(path, 'JPEGImages', data_folder, file.split('.')[0] + '.jpg')) 

    assert len(train_objects) == len(train_images)

    # save training JSON files and label map
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects}")
    print(f"File save path: {os.path.abspath(output_folder)}")

    assert len(val_objects) == len(val_images)
    # save val JSON files
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(val_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(val_objects, j)

    print(f"Total val images: {len(val_images)}")
    print(f"Total val objects: {n_objects_val}")
    print(f"File save path: {os.path.abspath(output_folder)}")

    assert len(test_objects) == len(test_images)

    # save test JSON files
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f"Total test images: {len(test_images)}")
    print(f"Total test objects: {n_objects_test}")
    print(f"File save path: {os.path.abspath(output_folder)}")

# Created for Coco Dataset-old
def parse_annotation_COCO(coco, imgId, catId):

    """
    Parses the annotations file to get the labels and bounding box coordinates, as well as difficulties
    """
    boxes = list()
    labels = list()
    difficulties = list()
    annotation_ids = coco.getAnnIds(imgIds=imgId, catIds=catId)
    anns = coco.loadAnns(annotation_ids)

    for ann in anns:
        cat_id=list()
        bbox = ann['bbox']
        cat_id.append(int(ann["category_id"]))
        label = coco.loadCats(cat_id)[0]['name']
        if label not in label_map:
            continue
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = xmin + int(bbox[2])
        ymax = ymin + int(bbox[3])

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(0)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

# Created for COCO Dataset-old
def create_data_lists_COCO(COCO_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param COCO_path: path to the 'COCO' images folder
    :param output_folder: folder where the JSONs must be saved
    """

    # COCO_path = os.path.abspath(COCO_path)
    ann_train_path = COCO_path+"/annotations/instances_train2017.json"
    # ann_test_path = COCO_path+"/annotations/instances_val2017.json"
    coco=COCO(ann_train_path)

    file_dict={}
    train_images = list()
    train_objects = list()
    n_objects_train = 0
    val_images = list()
    val_objects = list()
    n_objects_val = 0
    test_images = list()
    test_objects = list()
    n_objects_test = 0
    Image_ids = []

    for root, dirs, files in os.walk(COCO_path+"/images/train/"):
        path = root.split(os.sep) 
        folder = root.split('/')
        if(folder[4])=='':
            continue
        else:   
            category_id = coco.getCatIds([folder[4]])  
            for file in files:
                file_dict[file]=category_id

        np.random.shuffle(files)
        split1 = int(0.7 * len(files))
        split2 = int(0.9 * len(files))
        train_files = files[:split1]
        val_files = files[split1:split2]
        test_files = files[split2:] 
            
        for file in train_files: 
            image_id_temp = []
            image_id = int(file.split('.')[0].lstrip('0'))
            Image_ids.append(image_id)
            image_id_temp.append(image_id)

            objects = parse_annotation_COCO(coco, image_id_temp, file_dict[file])
            if len(objects['boxes']) == 0:
                continue
            n_objects_train += len(objects['boxes'])
            train_objects.append(objects)
            train_images.append(path[0]+'/'+file)

        for file in val_files: 
            image_id_temp = []
            image_id = int(file.split('.')[0].lstrip('0'))
            Image_ids.append(image_id)
            image_id_temp.append(image_id)

            objects = parse_annotation_COCO(coco, image_id_temp, file_dict[file])
            if len(objects['boxes']) == 0:
                continue
            n_objects_val += len(objects['boxes'])
            val_objects.append(objects)
            val_images.append(path[0]+'/'+file)
        
        for file in test_files: 
            image_id_temp = []
            image_id = int(file.split('.')[0].lstrip('0'))
            Image_ids.append(image_id)
            image_id_temp.append(image_id)

            objects = parse_annotation_COCO(coco, image_id_temp, file_dict[file])
            if len(objects['boxes']) == 0:
                continue
            n_objects_test += len(objects['boxes'])
            test_objects.append(objects)
            test_images.append(path[0]+'/'+file)

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects_train}")
    print(f"File save path: {os.path.abspath(output_folder)}")
    
    # Validation Data 
    assert len(val_objects) == len(val_images)
    # save val JSON files
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(val_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(val_objects, j)

    print(f"Total val images: {len(val_images)}")
    print(f"Total val objects: {n_objects_val}")
    print(f"File save path: {os.path.abspath(output_folder)}")

    # Test data
    # Image_ids = []
    # test_images = list()
    # test_objects = list()
    # n_objects_test = 0

    # coco=COCO(ann_test_path)
    # for root, dirs, files in os.walk(COCO_path+"/images/test/"):
    #     path = root.split(os.sep) 
    #     folder = root.split('/')
    #     if(folder[4])=='':
    #         continue
    #     else:   
    #         category_id = coco.getCatIds([folder[4]])  
    
    #     for file in files: 
    #         image_id_temp = []
    #         image_id = int(file.split('.')[0].lstrip('0'))
    #         Image_ids.append(image_id)
    #         image_id_temp.append(image_id)

    #         objects = parse_annotation_COCO(coco, image_id_temp, category_id)
    #         if len(objects['boxes']) == 0:
    #             continue
    #         n_objects_test += len(objects['boxes'])
    #         test_objects.append(objects)
    #         test_images.append(path[0]+'/'+file)

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f"Total test images: {len(test_images)}")
    print(f"Total test objects: {n_objects_test}")
    print(f"File save path: {os.path.abspath(output_folder)}")

# Created for COCO Dataset-used currently in training
def parse_annotation_COCO_new(filepath):
    
    """
    Parses the annotations file to get the labels and bounding box coordinates, as well as difficulties
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('path'):
        img_path = object.text

    for object in root.iter('object'):
        difficult = int(object.find('difficult').text)
    
        label = object.find('name').text.strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return img_path,{'boxes': boxes, 'labels': labels, 'difficulties':difficulties}

# Created for COCO Dataset-old
def create_data_lists_COCO_new(COCO_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param COCO_path: path to the 'COCO' images folder
    :param output_folder: folder where the JSONs must be saved
    """

    annotation_files=[]
    train_images = list()
    train_objects = list()
    n_objects_train = 0
    val_images = list()
    val_objects = list()
    n_objects_val = 0
    test_images = list()
    test_objects = list()
    n_objects_test = 0
    Image_ids = []


    for root, dirs, files in os.walk(COCO_path+"train/Annotations/"):
        for file in files:
            filepath = root+'/'+file
            annotation_files.append(filepath)
    print(len(annotation_files))
 
    # Split Data into Test, Train and Validation datasets - 70% train, 20%-validation, 10%-test
    np.random.shuffle(annotation_files) 
    split1 = int(0.7 * len(annotation_files))
    split2 = int(0.9 * len(annotation_files))
    train_files = annotation_files[:split1]
    val_files = annotation_files[split1:split2]
    test_files = annotation_files[split2:]

    for file in train_files: 
        imagepath, objects =  parse_annotation_COCO_new(file)
        if len(objects) == 0:
            continue
        n_objects_train += len(objects)
        train_objects.append(objects)
        train_images.append(imagepath)

    for file in val_files: 
        imagepath, objects =  parse_annotation_COCO_new(file)
        if len(objects) == 0:
            continue
        n_objects_val += len(objects)
        val_objects.append(objects)
        val_images.append(imagepath)
        
    for file in test_files: 
        imagepath, objects =  parse_annotation_COCO_new(file)
        if len(objects) == 0:
            continue
        n_objects_test += len(objects)
        test_objects.append(objects)
        test_images.append(imagepath)

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects_train}")
    print(f"File save path: {os.path.abspath(output_folder)}")
    
    # Validation Data 
    assert len(val_objects) == len(val_images)
    # save val JSON files
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(val_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(val_objects, j)

    print(f"Total val images: {len(val_images)}")
    print(f"Total val objects: {n_objects_val}")
    print(f"File save path: {os.path.abspath(output_folder)}")

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print(f"Total test images: {len(test_images)}")
    print(f"Total test objects: {n_objects_test}")
    print(f"File save path: {os.path.abspath(output_folder)}")

# Created for COCO Dataset-used currently in training
def get_continual_dataset(data_folder, dataset_name):
    """
    To retrieve data ffor training based on scenario for continual learning 
    from the csv file created initially when created the dataset.

    """
    
    COCO_path= './filtered_coco_dataset_2017/'
    data = []
    print('Loading Dataset : ', dataset_name)    
    df = pd.read_csv(data_folder+"datasets.csv")
    df_new  = df.loc[df['Dataset'] == dataset_name]
    for i in df_new.iterrows():
        data.append(COCO_path + 'train/Annotations/'+ i[1][2] + '/' + i[1][1])
    create_continual_data_lists_COCO_new(COCO_path=COCO_path, output_folder='./output/cl/', datasets = data)

# Created for COCO Dataset continual-used currently in training
def create_continual_data_lists_COCO_new(COCO_path, output_folder, datasets):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param COCO_path: path to the 'COCO' images folder
    :param output_folder: folder where the JSONs must be saved
    """

    annotation_files=[]
    train_images = list()
    train_objects = list()
    n_objects_train = 0
    val_images = list()
    val_objects = list()
    n_objects_val = 0
    test_images = list()
    test_objects = list()
    n_objects_test = 0
    Image_ids = []


    # fileObject = open(output_folder+"cl_data.json", "r")
    # jsonContent = fileObject.read()
    # d=json.loads(jsonContent)
    # D1 = datasets[0]
    # D2 = datasets[1]
    np.random.shuffle(datasets)
    split1 = int(0.7 * len(datasets))
    # split2 = int(0.9 * len(annotation_files))
    train_files = datasets[:split1]
    val_files = datasets[split1:]
    # test_files = annotation_files[split2:]

    # for j in i:
    #     print(j)

    for file in train_files: 
        imagepath, objects =  parse_annotation_COCO_new(file)
        if len(objects) == 0:
            continue
        n_objects_train += len(objects)
        train_objects.append(objects)
        train_images.append(imagepath)

    for file in val_files: 
        imagepath, objects =  parse_annotation_COCO_new(file)
        if len(objects) == 0:
            continue
        n_objects_val += len(objects)
        val_objects.append(objects)
        val_images.append(imagepath)
        
    # for file in test_files: 
    #     imagepath, objects =  parse_annotation_COCO_new(file)
    #     if len(objects) == 0:
    #         continue
    #     n_objects_test += len(objects)
    #     test_objects.append(objects)
    #     test_images.append(imagepath)

    # assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print(f"Total training images: {len(train_images)}")
    print(f"Total training objects: {n_objects_train}")
    print(f"File save path: {os.path.abspath(output_folder)}")
    
    # Validation Data 
    assert len(val_objects) == len(val_images)
    # save val JSON files
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(val_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(val_objects, j)

    print(f"Total val images: {len(val_images)}")
    print(f"Total val objects: {n_objects_val}")
    print(f"File save path: {os.path.abspath(output_folder)}")

    # assert len(test_objects) == len(test_images)

    # # Save to file
    # with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
    #     json.dump(test_images, j)
    # with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
    #     json.dump(test_objects, j)

    # print(f"Total test images: {len(test_images)}")
    # print(f"Total test objects: {n_objects_test}")
    # print(f"File save path: {os.path.abspath(output_folder)}")

# Eval function created separtely for training, includes creating log files
def evaluate_train(test_loader, model, log_path, filename, epoch):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            #print("images")
            #print(images.size());
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            #print( predicted_locs, predicted_scores)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            
        # Calculate mAP
        APs, mAP, Prec, Rec = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.7)

    # Print AP for each class
    pp.pprint(APs)

    # Creating log files
    log_path = log_path + "APs\\"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # print(filename)
    file = filename.split('_logs_')
    # print(file)
    filename = "Eval_logs_"+file[1]
    filepath = log_path + '/' + filename
    if not os.path.isfile(filepath):
        print('\nFile Doesn\'t Exist, creating file')
        f_string = '\nEpoch\t'
        for label in coco_labels:
            f_string = f_string + label + '\t'
        f_string = f_string = f_string + 'mAP\t'    
        f_string = f_string = f_string + 'P\t'
        f_string = f_string = f_string + 'R'
        with open(file=filepath, mode='w') as f:
            f.write(f_string)

    # Rounding off to the third digit 
    f_string = '\n' + str(epoch) + '\t'
    for label in coco_labels:
        f_string = f_string + str(round(APs[label],3)) + '\t'        
    f_string = f_string = f_string + str(round(mAP,3)) + '\t'
    f_string = f_string = f_string + str(round(Prec,3)) + '\t'
    f_string = f_string = f_string + str(round(Rec,3))
    with open(file=filepath, mode='a+') as f:
        f.write(f_string)


    print('\nMean Average Precision (mAP): %.3f' % mAP)
    print('\nPrecision (P): %.3f' % Prec)
    print('\nRecall (R): %.3f \n' % Rec)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            #print("images")
            #print(images.size());
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            #print( predicted_locs, predicted_scores)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            
        # Calculate mAP
        APs, mAP, Prec, Rec = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.7)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    print('\nPrecision (P): %.3f' % Prec)
    print('\nRecall (R): %.3f \n' % Rec)


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, iou_threshold):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    #print(len(det_boxes),len(det_labels),len(det_scores),len(true_boxes),(true_labels),len(true_difficulties))
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)
    true_boxes=[b.to(device) for b in true_boxes]
    true_labels=[l.to(device) for l in true_labels]
    true_difficulties=[d.to(device) for d in true_difficulties]
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        #print (true_labels[i].size(0))
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        #print (det_labels[i])
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0).to(device)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    prec = torch.zeros((n_classes - 1), dtype=torch.float) 
    rec = torch.zeros((n_classes - 1), dtype=torch.float)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        tp=0
        fp=0
        #print("n_class_detection"+str(n_class_detections))
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0))).to(torch.device('cuda'))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'
            
            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > iou_threshold:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1 
                        tp=tp+1 # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
                        fp=fp+1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1
                fp=fp+1
        #print("TP::"+str(tp))
        #print("FP::"+str(fp))
        #print("no_of objects::"+str(n_easy_class_objects))

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)
        #print(cumul_true_positives)
        prec[c-1]=cumul_precision.mean()
        rec[c-1]= cumul_recall.mean()
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()
    mean_prec=prec.mean().item()
    mean_rec=rec.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    prec= {rev_label_map[c + 1]: v for c, v in enumerate(prec.tolist())}
    rec= {rev_label_map[c + 1]: v for c, v in enumerate(rec.tolist())}


    return average_precisions, mean_average_precision, mean_prec, mean_rec


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)
    

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    #print(boxes,old_dims)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST','VAL'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chancsse - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
           new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                       new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
           new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    #t=torch.cat((new_image,new_image),dim=0)
    #print(t.size())
    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer, min_loss, checkpoint):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'min_loss': min_loss}
    filename = checkpoint
    torch.save(state, filename,_use_new_zipfile_serialization=False)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
