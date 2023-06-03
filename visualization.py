import pandas as pd
import os
import matplotlib.pyplot as plt
import random


log_path = "./logs/cl_logs/train_logs/"                                                                  # Log files path
log_files =[]                                                                                            # List of Log files                               
d_content = os.listdir(log_path)
for file in d_content:
    if '.txt' in file:
        log_files.append(log_path+file)
sorted_by_mtime_descending = sorted(log_files, key=lambda t: -os.stat(t).st_mtime)                       # Sorting Log files by latest modified

# Display for latest modified log file 
filepath = sorted_by_mtime_descending[0]

# to display for a particular file 
# filepath = log_path + "train_logs_m2_ewc_test_2_01_05_2023-15_11_38.txt"


get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]                         # Selecting Random colours for Graph visualisation


df = pd.read_csv(filepath, sep="\t",index_col=False)                                                     # Reading the log file and selecting required columns 
columns = ['Val Loss Average', 'Loss Average']

filename = (filepath.split('/')[4]).split('_')[2]

                         
# Plotting the graph of values vs No. of Epochs 
ax = plt.gca() 
df.plot(
        title=filename,
        kind = 'line',
        x = 'Epoch',
        y = columns,
        color = get_colors(len(columns)),ax = ax,
        ylim=(0,max(df['Val Loss Average'].max()+2, df['Loss Average'].max()+2)))

plt.show()