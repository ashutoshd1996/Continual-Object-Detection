import wget
import zipfile
import os, sys
from PIL import Image
from coco_dataset import coco_dataset_download as cocod
import requests
from pycocotools.coco import COCO
import time

"""
Old function to download data 

"""
def get_data():
  wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', '../annotations_trainval2017.zip')
  with zipfile.ZipFile('../annotations_trainval2017.zip', 'r') as zip_ref:
      zip_ref.extractall('../')
  os.remove("../annotations_trainval2017.zip")
  

def get_images(method):

  COCO_path = './COCO/images'
  class_names=['person', 'car', 'stop sign','traffic light', 'motorcycle',  'boat', 'truck','bus','airplane','suitcase']
  if method=='train':
    annotations_path='./COCO/annotations/instances_train2017.json'
    image_count=450
  elif method=='test':
    annotations_path='./COCO/annotations/instances_val2017.json'
    image_count=50
  else: 
    print('Enter Test or Train')
    return

  images_path = COCO_path + '/' +method
  if not os.path.exists(images_path):
    os.mkdir(images_path)
  coco = COCO(annotations_path)

  for class_name in class_names:
    count=0 # count to check no of images for each class name 

    # if method=='train':
    save_path = images_path+'/'+class_name
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    # else:
    #     save_path = images_path

    catIds = coco.getCatIds(catNms=[class_name])
    # Get the corresponding image ids and images using loadImgs
    imgIds = coco.getImgIds(catIds=catIds)
    
    
    for imgId in imgIds:
      exists = 0
      img = coco.loadImgs([imgId])  
      annotation_ids = coco.getAnnIds(imgIds=[imgId], catIds=catIds)
      anns = coco.loadAnns(annotation_ids)
      n_objects = len(anns)  
      if n_objects < 4:      
        img_data = requests.get(img[0]['coco_url']).content
        for root, dirs, files in os.walk(images_path):
          if img[0]['file_name'] in files:
            exists = 1
            break
        if exists == 0:    
          with open(save_path +'/'+ img[0]['file_name'], 'wb') as handler:
              handler.write(img_data)
          count+=1
          print('no.of image:',count)
          if count>=image_count:
              print('finished images download')
              break          
        else:
          continue
      else: 
        continue   

get_data()
get_images('train')
