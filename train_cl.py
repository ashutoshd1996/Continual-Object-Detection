import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import COCODataset
from utils import *
from datetime import datetime
from torchsummary import summary

# Data parameters
log_path = './logs/'
filename =''
filepath=''
data_folder = './output/'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
ewc_lambda = 0.1   # EWC Regularizer value
batch_size = 4  # batch size
iterations = 40000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 500 # print training status every __ batches
lr = 1e-3  # learning rate      
decay_lr_at = [25000, 30000, 35000]  # decay learning rate after these many iterations Experiment1ye
decay_lr_to = 0.1 # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay 
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
min_loss=1000000000000000000.0

cudnn.benchmark = True

def main():

    global decay_lr_at, decay_lr_to, label_map, ewc_lambda, ewc_check, min_loss, lr, iterations
    global filename, filepath, log_path, dt_string, log_path, data_folder
    global start_epoch
    global fisher_matrix, opt_params
    fisher_matrix = {}                          # Dictionary to store the EWC fisher matrix
    opt_params = {}                             # Dictionary to store the EWC optimal parameters of previous tasks / datasets in our case        

    model_name = None

    cl = input("Continual Learning ? ( Yes / No ) : ") #

    if cl.lower() == "yes":

        # Creating seperate path for continual learning approaches
        log_path = log_path + 'cl_logs/'
        data_folder = data_folder + 'cl/'

        data_tasks = 7                              # Number of tasks / datasets to train the model on 
        ewc_check = input('Want to use EWC ? : ')   # Enter if EWC is to be added to the model

        # Continuously train on all continual learning datasets
        for i in range(1, data_tasks):
            
            # Loading the Datasets 
            dataset_name = "D"+str(i+1)                   
            get_continual_dataset(data_folder, dataset_name.upper())

            # Create Model Name based on which dataset is getting loaded
            model_name_str = "M"+str(i+1)
            
            # If ewc is enabled then add _ewc to the model name string
            if(ewc_check.lower() == "yes"):
                model_name_str += "_ewc"

            # Creating Directory for Logs, if it doesn't already exist
            if not os.path.exists(log_path):
                os.mkdir(log_path)

            # Create Train log file with timestamp
            dt = datetime.now()
            dt_string = dt.strftime("%d_%m_%Y-%H_%M_%S")
            filename = 'train_logs_'+model_name_str+'_'+dt_string+'.txt'
            filepath = log_path + '/train_logs/'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            filepath = filepath + filename
            if not os.path.isfile(filepath):
                print('\nFile Doesn\'t Exist, creating file')
                with open(file=filepath, mode='w') as f:
                    f.write('Epoch\tTrain Loader size\tVal Loss\tVal Loss Average\tBatch Time\tBatch Time Average\tData Time\tData Time Average\tLoss\tLoss Average')

            # creating path to save the model
            model_name = './models/'+ model_name_str + '.pth.tar'

            start_epoch = 0
            
            if i == 0:
                model = SSD300(n_classes=n_classes)
                        
            # If it is not the first Iteration then load the previous model 
            else:
                prev_model_str = list(model_name_str)
                prev_model_str[1] = str(i) 
                prev_model_str = ''.join(prev_model_str)
                prev_model_name = './models/'+ prev_model_str + '.pth.tar'
                prev_model = torch.load(prev_model_name,map_location=device)
                start_epoch = prev_model['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
                model = prev_model['model']
                optimizer = prev_model['optimizer']
                # min_loss = prev_model['min_loss']            
                # print(summary(model, input_size=(3, 300, 300), verbose=2))

            model = model.to(device)

            biases = list() # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
            not_biases = list()

            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    if param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        not_biases.append(param)
                        
            optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                        lr=lr, momentum=momentum, weight_decay=weight_decay)


            criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

            # Creating custom dataloaders
            train_dataset = COCODataset(data_folder,
                                            split='train',
                                            keep_difficult=keep_difficult)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    collate_fn=train_dataset.collate_fn, num_workers=workers,
                                                    pin_memory=True)  # note that we're passing the collate function here
            val_dataset = COCODataset(data_folder,split='val')
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                    collate_fn=val_dataset.collate_fn, num_workers=workers,
                                                    pin_memory=True)  # using `collate_fn()` here

            epochs = iterations // (len(train_dataset) // batch_size)
            
            if  epochs > 200:   # clip number of epochs to max 120 
                epochs = 200
                new_iterations = epochs * (len(train_dataset) // batch_size) # Re-calculate number of iterations

                decay_lr_at = [int ((j/ iterations) * new_iterations) for j in decay_lr_at] # Re-scale at which iteration to-perform lr-decay
                
                iterations = new_iterations
         
            decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

            start_epoch = 0

            # Creating log for the hyperparameters
            hp_name = 'Hyperparameters_'+(model_name.split('/')[2]).split('.')[0]+'_'+dt_string+'.txt'
            hp_path = log_path + '/Hyperparameters/'
            if not os.path.exists(hp_path):
                os.mkdir(hp_path)
            hp_path = hp_path + hp_name
            if not os.path.isfile(hp_path):
                with open(file=hp_path, mode='w') as f:
                    f.write('Epochs\tTrain dataset Size\tVal dataset size\tBatch size\tIterations\tLearning Rate\tDecay LR factor\tWeight Decay')
                    if ewc_check.lower() == "yes":    # If EWC is enabled then add the additional EWC lambda parameter column
                        f.write('\tEWC Regularizer')   

            with open(file=hp_path, mode='a+') as f:   
                if ewc_check.lower() == "yes":         # If EWC is enabled then add the additional EWC lambda parameter value
                    f.writelines('\n{0}\t{1}\t'
                        '{2}\t{3}\t'
                        '{4}\t{5}\t'
                        '{6}\t{7}\t'
                        '{8}\t'.format(epochs, len(train_dataset), len(val_dataset),batch_size, iterations,
                                                                        lr, decay_lr_to, weight_decay, ewc_lambda))
                else :    
                    f.writelines('\n{0}\t{1}\t'
                            '{2}\t{3}\t'
                            '{4}\t{5}\t'
                            '{6}\t{7}\t'.format(epochs, len(train_dataset), len(val_dataset),batch_size, iterations,
                                                                            lr, decay_lr_to, weight_decay))
                
            print('Hyperparameters Logged')

            print('\nTraining for %d epochs.\n' % epochs)
            
            # EWC initialization if it is enabled
            if ewc_check.lower() == "yes":
                
                fisher_path = log_path + '/Fisher_Matrix/'  # Path to store the Fisher Matrix 
                
                if not os.path.exists(fisher_path):
                    os.mkdir(fisher_path)

                if (i!=0):
                    # If using pre=trained model on previous dataset, we are going to load the fisher matrix
                    fisher_load_name = prev_model_str + "_fisher.pt"
                    opt_params_load_name =  prev_model_str + "_opt.pt"
                    fisher_load_path = fisher_path + fisher_load_name
                    opt_load_path = fisher_path + opt_params_load_name
                    if os.path.isfile(fisher_load_path) or os.path.isfile(opt_load_path):
                        fisher_matrix, opt_params = load_fisher(fisher_load_path, opt_load_path)
                    else:
                        print("Fisher Matrix or Optimal Params Matrix not found for : ", fisher_load_name, " , ", opt_params_load_name)
                        exit(1)
                
                fisher_name = (model_name.split('/')[2]).split('.')[0]+"_fisher.pt"
                opt_name = (model_name.split('/')[2]).split('.')[0]+"_opt.pt"

                opt_params_path = fisher_path + opt_name
                fisher_path = fisher_path + fisher_name                
                print("EWC Lambda Value is : ", ewc_lambda)

            # Epochs
            for epoch in range(start_epoch, epochs):

                # Decay learning rate at particular epochs
                if epoch in decay_lr_at:
                    adjust_learning_rate(optimizer, decay_lr_to)

                # One epoch's training
                train(train_loader=train_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loader=val_loader,
                    modelname=model_name, 
                    task_id=i)

            # If EWC is enabled update the fisher matrix to add to the current task and save it to the specified path 
            if ewc_check.lower() == "yes":
                update_fisher(model, train_loader, device, criterion, optimizer, i)
                save_fisher(fisher_matrix, fisher_path, opt_params, opt_params_path)
            
            # To continuosly run, created till end of scenarios
            if i!= data_tasks - 1: 
                check = input('Do you want to continue ? : ')
                if check.lower() != "yes":
                    exit(0)
                else:
                    min_loss=1000000000000000000.0
                    uparams = input ("Do you want to update hyperparameters ? : ")
                     
                    if uparams.lower() == "yes":
                        lr = float(input("Enter new learning Rate : "))
                        if ewc_check.lower() == "yes":
                            ewc_lambda = float(input("Enter new EWC Lambda Value : "))
                    else:
                        lr = 1e-3  # learning rate
                    iterations = 40000       
                    decay_lr_at = [25000, 30000, 35000] 
                    
    
    # If its not continual learning, code runs the section below
    else:
        
        check = "yes"
        checkpoint = None

        #  to run code until user decides 
        while check.lower() == "yes":
            
            model_name_str = input("\nEnter Model name : ")

            # Create dataset for training
            create_data_lists_COCO_new(COCO_path='./filtered_coco_dataset_2017/',   
                        output_folder='./output') 

            # Create Training Log file 
            dt = datetime.now()
            dt_string = dt.strftime("%d_%m_%Y-%H_%M_%S")
            filename = 'train_logs_'+model_name_str+'_'+dt_string+'.txt'
            filepath = log_path + '/train_logs/'
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            filepath = filepath + filename
            if not os.path.isfile(filepath):
                print('\nFile Doesn\'t Exist, creating file')
                with open(file=filepath, mode='w') as f:
                    f.write('Epoch\tTrain Loader size\tVal Loss\tVal Loss Average\tBatch Time\tBatch Time Average\tData Time\tData Time Average\tLoss\tLoss Average')

            model_name = './models/'+ model_name_str + '.pth.tar'

            # if new model, i.e. starting from Scratch
            if checkpoint is None:
                start_epoch = 0
                model = SSD300(n_classes=n_classes)
                # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
                biases = list()
                not_biases = list()
                for param_name, param in model.named_parameters():
                    if param.requires_grad:
                        if param_name.endswith('.bias'):
                            biases.append(param)
                        else:
                            not_biases.append(param)
                    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                            lr=lr, momentum=momentum, weight_decay=weight_decay)

            # To retrain existing model
            else:
                # Load model
                checkpoint = torch.load(model_name)     
                start_epoch = checkpoint['epoch'] + 1
                print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
                model = checkpoint['model']
                optimizer = checkpoint['optimizer']

            model = model.to(device)
            criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

            # Custom dataloaders
            train_dataset = COCODataset(data_folder,
                                            split='train',
                                            keep_difficult=keep_difficult)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    collate_fn=train_dataset.collate_fn, num_workers=workers,
                                                    pin_memory=True)  # note that we're passing the collate function here
            val_dataset = COCODataset(data_folder,split='val')
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                    collate_fn=val_dataset.collate_fn, num_workers=workers,
                                                    pin_memory=True)  # using `collate_fn()` here

            epochs = iterations // (len(train_dataset) // batch_size)
            decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

            print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)


            # Log Hyperparameters
            hp_name = 'Hyperparameters_'+(model_name.split('/')[2]).split('.')[0]+'_'+dt_string+'.txt'
            hp_path = log_path + '/Hyperparameters/'
            if not os.path.exists(hp_path):
                os.mkdir(hp_path)
            hp_path = hp_path + hp_name
            if not os.path.isfile(hp_path):
                with open(file=hp_path, mode='w') as f:
                    f.write('Epochs\tTrain dataset Size\tVal dataset size\tBatch size\tIterations\tLearning Rate\tDecay LR factor\tWeight Decay')

            with open(file=hp_path, mode='a+') as f:   
                f.writelines('\n{0}\t{1}\t'
                        '{2}\t{3}\t'
                        '{4}\t{5}\t'
                        '{6}\t{7}\t'.format(epochs, len(train_dataset), len(val_dataset),batch_size, iterations,
                                                                        lr, decay_lr_to, weight_decay))
            print('Hyperparameters Logged')

            print('\nTraining for %d epochs.\n' % epochs)

            # Epochs
            for epoch in range(start_epoch, epochs):

                # Decay learning rate at particular epochs
                if epoch in decay_lr_at:
                    adjust_learning_rate(optimizer, decay_lr_to)

                # One epoch's training
                train(train_loader=train_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loader=val_loader,
                    modelname=model_name)
            
            # User's input for retraining
            check = input("Do you want to continue ? :")

def train(train_loader, model, criterion, optimizer, epoch, val_loader, modelname, task_id = 0):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    global min_loss
    global fisher_matrix
    global opt_params
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    val_losses = AverageMeter() #val loss
    start = time.time()
    start_data = time.time()
    # Batches
    print("Started training epoch : "+str(epoch))

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
		
        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Reset Gradients
        optimizer.zero_grad()

        # If EWC is enabled then calculate ewc loss and add it to the train loss
        if ewc_check.lower() == "yes":
            if task_id != 0:
                for task in range(0, task_id):
                    for name, param in model.named_parameters():
                        fisher = fisher_matrix[task][name]
                        optpar = opt_params[task][name]
                        fisher_penalty = (fisher * (optpar - param).pow(2)).sum()
                        ewc_loss = fisher_penalty * ewc_lambda
                        loss += ewc_loss
        
        # Backward prop
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        
        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()



    print("Started validating for epoch : "+str(epoch))   
    # free some memory since their histories may be stored
    for i, (images, boxes, labels, _) in enumerate(val_loader):
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # loss
        val_loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar


        val_losses.update(val_loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
    data_time.update(time.time() - start_data)



    # print status
    print("================================================================================================================================================")
    print(val_losses)
    print("================================================================================================================================================")
    print('Epoch: [{0}][{1}]\t'
            'Val Loss {val_loss.val:.2f} ({val_loss.avg:.2f})\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(epoch,len(train_loader),val_loss=val_losses,
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses))
    
    # logging into train.txt
    with open(file=filepath, mode='a+') as f:
        f.writelines('\n{0}\t{1}\t'
            '{val_loss.val:.2f}\t{val_loss.avg:.2f}\t'
            '{batch_time.val:.3f}\t{batch_time.avg:.3f}\t'
            '{data_time.val:.4f}\t{data_time.avg:.4f}\t'
            '{loss.val:.5f}\t{loss.avg:.5f}\t'.format(epoch, len(train_loader), val_loss=val_losses,
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses))


    # save checkpoint after each epoch
    if val_losses.avg<min_loss:
        min_loss=val_losses.avg
        save_checkpoint(epoch, model, optimizer, min_loss, modelname)
        print('CheckPoint saved for minimum loss:: {val_loss.avg:.2f}\t'.format(val_loss=val_losses))
        evaluate_train(val_loader,model,log_path, filename, epoch)
    print(val_losses.avg)########################################################################################################################################################################################
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def save_fisher(fisher_matrix, fisher_path, opt_params, opt_path):

    # Save the fisher matrix of the current task to the specified path
    print("Saving fisher matrix to : ", fisher_path)         
    torch.save(fisher_matrix, fisher_path)

    # Save the optimal parameters of the current task to the specified path
    print("Saving Parameters to : ", opt_path)         
    torch.save(opt_params, opt_path)

def load_fisher(fisher_path, opt_path):

    # Load the fisher matrix of the current task from the specified path
    print("\nLoading fisher matrix from : ", fisher_path)
    fisher_matrix = torch.load(fisher_path)
    
    # Load the optimal parameters of the current task from the specified path
    print("\nLoading Parameters from : ", opt_path)
    opt_params = torch.load(opt_path)
    return fisher_matrix, opt_params

def update_fisher(model, dataloader, device, criterion, optimizer, task_id):
    
    """

    Function to create and update the fisher matrix and optimal parameters of a task specified by task_id

    """
    global fisher_matrix
    global opt_params

    print("\nUpdating fisher matrix for task : ", task_id+1)
    
    model.train()
    optimizer.zero_grad()       # Resetting Gradients

    # Calulating the loss and gradients 
    for i, (images, boxes, labels, _) in enumerate(dataloader):
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        loss.backward()     
       
    fisher_matrix[task_id] = {}  
    opt_params[task_id] = {}

  # gradients are used to calculate the fisher matrix and optimal parameters
    for name, param in model.named_parameters():    
        opt_params[task_id][name] = param.data.clone()
        fisher_matrix[task_id][name] = param.grad.data.clone().pow(2)

if __name__ == '__main__':
    main()       