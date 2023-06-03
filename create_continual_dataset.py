import os
import random 
import utils
import pandas as pd
import time


# creating dataset based on scenarios
# storing this dataset into a dataframe and saving it as a csv file

annotation_files=[]

# data path 
COCO_path='./filtered_coco_dataset_2017/'

# output path to save the csv file 
output_path = './output/cl/'

coco_labels = ('person', 'car', 'stop sign','traffic light', 'motorcycle','boat', 'truck','bus','airplane','suitcase')


# adding the annotation files into dictionary 
for root, dirs, files in os.walk(COCO_path+"train/Annotations/"):
    for file in files:
        filepath = root+'/'+file
        annotation_files.append(filepath)


# Percentage of old data in new dataset
pct = 0.1

# Create the pandas DataFrame with column name is provided explicitly
# Dataframe contains name of annotation xml file of each image as Data
df = pd.DataFrame(columns=["Dataset", "Data", "Class"])
  
# Creating a list for each scenario
D1 = []
D2 = []
D3 = []
D4 = []
D5 = []
D6 = []
D7 = []

# temporary list for adding blend of old data
T1 = []
T2 = []
T3 = []
T4 = []
T5 = []
T6 = []

# Dataset D1
for i in coco_labels[0:4]:
    D = []
    for item in annotation_files:
        if i in item:
            D.append(item)
            df_new_row = pd.DataFrame({"Dataset":"D1", "Data":item.split('/')[5], "Class":i},index=["Data"])
            df = pd.concat([df, df_new_row],ignore_index=True)
    T1=T1+random.sample(D, int(pct*len(D)))
    T2=T2+random.sample(D, int(pct*len(D)))
    T3=T3+random.sample(D, int(pct*len(D)))
    T4=T4+random.sample(D, int(pct*len(D)))
    T5=T5+random.sample(D, int(pct*len(D)))
    T6=T6+random.sample(D, int(pct*len(D)))  
    D1 = D1 + D

print(len(D1))  

# Dataset D2   
for i in coco_labels[4:7]:
    D = []
    count = 0
    for item in annotation_files:
        if i in item and count < 300:
            D.append(item)
            count +=1
    T2=T2+random.sample(D, int(pct*len(D)))
    T3=T3+random.sample(D, int(pct*len(D)))
    T4=T4+random.sample(D, int(pct*len(D)))
    T5=T5+random.sample(D, int(pct*len(D))) 
    T6=T6+random.sample(D, int(pct*len(D)))        
    D2 = D2 + D

D2_full = D1 + D2
D2 = D2 + T1
print(len(D2))


# Dataset D3
for i in coco_labels[4:6]:
    D = []
    count = 0
    for item in annotation_files:
        if i in item and count < 100 and item not in D2_full:
            D.append(item) 
            count +=1  
    T3=T3+random.sample(D, int(pct*len(D)))
    T4=T4+random.sample(D, int(pct*len(D)))
    T5=T5+random.sample(D, int(pct*len(D)))
    T6=T6+random.sample(D, int(pct*len(D)))                 
    D3 = D3 + D
D3_full = D2_full + D3
D3 = D3 + T2
print(len(D3))



# Dataset D4 
for i in coco_labels[4:7]:  
    D = []
    if i == coco_labels[6]:
        count_1 = 0
        for item in annotation_files:
            if (i in item) and (count_1 < 200) and (item not in D3_full):
                D.append(item) 
                count_1 +=1   
                
    else:
        count_2 = 0
        for item in annotation_files:
            if i in item and count_2 < 100 and item not in D3_full:
                D.append(item) 
                count_2 +=1
    
    T4=T4+random.sample(D, int(pct*len(D)))
    T5=T5+random.sample(D, int(pct*len(D)))
    T6=T6+random.sample(D, int(pct*len(D)))                
    D4 = D4 + D    

D4_full = D3_full + D4
D4 = D4+ T3
print(len(D4))
# print(len(D4_full))


# Dataset D5 
i = coco_labels[7]
D = []
for item in annotation_files:
    if i in item:
        D.append(item)
T5=T5+random.sample(D, int(pct*len(D)))      
T6=T6+random.sample(D, int(pct*len(D)))  

D5_full = D4_full + D
D5 = D+ T4
print(len(D5))
# print(len(D5_full))


# Dataset D6 
i = coco_labels[8]
D = []
for item in annotation_files:
    if i in item:
        D.append(item)
T6=T6+random.sample(D, int(pct*len(D)))      

D6_full = D5_full + D
D6 = D + T5
print(len(D6))
# print(len(D6_full))


# Dataset D7
i =  coco_labels[9]
D = []
for item in annotation_files:
    if i in item:
        D.append(item)

D7_full = D6_full + D
D7 = D + T6
print(len(D7))
# print(len(D7_full))

# adding the created datsets into the dataframe
for i in D2:
    df_new_row = pd.DataFrame({"Dataset":"D2", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
for i in D3:
    df_new_row = pd.DataFrame({"Dataset":"D3", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
for i in D4:
    df_new_row = pd.DataFrame({"Dataset":"D4", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
for i in D5:
    df_new_row = pd.DataFrame({"Dataset":"D5", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
for i in D6:
    df_new_row = pd.DataFrame({"Dataset":"D6", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
for i in D7:
    df_new_row = pd.DataFrame({"Dataset":"D7", "Data":i.split('/')[5], "Class":i.split('/')[4]},index=["Data"])
    df = pd.concat([df, df_new_row],ignore_index=True)
    

# Saving the data
df.to_csv(output_path+"datasets.csv", index=False)



"""

To read data from csv file, copy or uncomment below code

"""
# df = pd.read_csv(output_path+"datasets.csv")
# df_new  = df.loc[df['Dataset'] == 'D2']
# # list_data = df_new['Data'].to_list()
# data = []
# for i in df_new.iterrows():
#     data.append(COCO_path + 'train/Annotations/'+ i[1][2] + '/' + i[1][1])
# print(len(data))

