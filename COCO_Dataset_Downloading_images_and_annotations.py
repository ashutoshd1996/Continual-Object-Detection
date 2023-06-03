from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import json
from pascal_voc_writer import Writer
import os
import wget
import zipfile

# Downloading COCO dataset annotations file
"""
Uncomment to download data 
"""
# wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', '../annotations_trainval2017.zip')
# with zipfile.ZipFile('../annotations_trainval2017.zip', 'r') as zip_ref:
#       zip_ref.extractall('./COCO/annotations')

# Annotation file type, train or test data to be specified
dataType='train2017'

# number of images per class
max_count=500

# number of objects
max_objects=10

# train data directory
tarDir='filtered_coco_dataset_2017/train'
iscrowd=None
annFile='./COCO/annotations/instances_{}.json'.format(dataType)

coco=COCO(annFile)

# class names
catNameList=['person', 'car', 'stop sign','traffic light', 'motorcycle',  'boat', 'truck','bus','airplane','suitcase']

# creating annotation xml file for each image and downloading the images
for t in range(10):
    catName=catNameList[t]
    catIds = coco.getCatIds(catNms=[catName]);
    imgIds = coco.getImgIds(catIds=catIds);
    imgIds = coco.getImgIds(imgIds = imgIds)
    filteredImgIds = []

    # filetering images based on number of objects in image
    for imgId in imgIds:
        annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=iscrowd)
        if len(annIds)<max_objects:
            filteredImgIds.append(imgId)

    filteredImgIds=random.sample(filteredImgIds, max_count)
    print(len(filteredImgIds))

    # Downloading the images
    coco.download(tarDir+'/JPEGImages'+'/'+catName+'/',imgIds=filteredImgIds)
    annIds1=coco.getAnnIds(imgIds=filteredImgIds, iscrowd=None)
    anns1=coco.loadAnns(annIds1)

    # creating xml file for each image
    for imgId in filteredImgIds:
        annIds = coco.getAnnIds(imgIds=[imgId], catIds=catIds)
        cats = coco.loadCats(catIds)
        cat_idx = {}
        for c in cats:
            cat_idx[c['id']] = c['name']

        if len(annIds) > 0:
            img_fname = coco.imgs[imgId]['file_name']
            print(img_fname)
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            
            # writing image size
            writer = Writer(img_fname, coco.imgs[imgId]['width'], coco.imgs[imgId]['height'])
            writer.template_parameters['path']='./'+tarDir+'/JPEGImages'+'/'+catName+'/'+img_fname
        
            anns = coco.loadAnns(annIds)
            for a in anns:
                # writing bounding box coordinates for each object in the image
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]

                # writing class name
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3])))
                try:
                    writer.save(tarDir+'/Annotations/'+catName+'/'+label_fname)
                except FileNotFoundError:
                    if not os.path.exists(tarDir+'/Annotations'):
                        os.mkdir(tarDir+'/Annotations')
                    os.mkdir(tarDir+'/Annotations/'+catName)
                    writer.save(tarDir+'/Annotations/'+catName+'/'+label_fname)
