# Continual_Object_Detection
MSc. Master Project 

Applying continual learning approaches to improve the reliability of Object Detection models in the industrial environment

contains the following steps:
• Data Processing
• Implementing continual learning methods on a base object detection model
• Comparison and analysis of the results

Repository Organisation 
├── logs                                                      <- Contains all log files
├── output                                                    <- Contains the Dataset files 
├── COCO_Dataset_Downloading_images_and_annotations.py        <- Dowloading Images and creating annotation xml file for each image
├── create_continual_dataset.py                               <- Creating dataset for continual learning
├── create_data_lists.py                                      <- Preparing Dataset for training
├── datasets.py                                               <- Creating custom datasets to be used by the dataloaders
├── detect.py                                                 <- Detect objects using trained model
├── eval.py                                                   <- Evaluate performance trained model
├── getData.py                                                <- Old way of downloading data
├── model.py                                                  <- Model used for training
├── train_cl.py                                               <- Training model for different scenarios 
├── utils.py                                                  <- Contains all functions to be called during training and evaluation
├── visualization.py                                          <- Visualize results of training
   
### Model Training
1. To download data run :
   ```bash python COCO_Dataset_Downloading_images_and_annotations.py ```
2. If you want to use continual learning run : 
   ```bash python create_continual_datasets.py ```
3. To train model run : 
   ```bash python train_cl.py ```  
4. To visualize training results : 
   ```bash python visualization.py ```
5. To ealuate performance of model : 
   ```bash python eval.py
6. To detect objects/test model :
   ```bash python detect.py


### Citation

If you found this work useful, feel free to cite it as

```
@misc{Ashutosh2023CL,
  title={Applying continual learning approaches to improve the reliability of Object Detection models in the industrial environment},
  author={Ashutosh Dinesh},
  year={2023},
  publisher={DFKI, RPTU Kaiserslautern, Master Project}
}
```


