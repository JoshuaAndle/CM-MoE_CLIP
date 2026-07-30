import os
import torch
from torchvision import datasets,transforms
import copy
import math


########################################################################################################################################
###  Online Datastreams
########################################################################################################################################


### Switches between PMNIST and CIFAR-100 splits, focusing on training on several different tasks (distributions)
def get_online_mixed_cifar_pmnist(set_num:int = 0, split:str = "train"):
    
    if os.path.isfile(("./data/online_split_cifar/0/train/set_" + str(set_num) + "/X.pt")) == False:
        print("No online dataset detected for cifar subsets. Creating new sets prior to loading task.")
        make_online_splitcifar()
        
    if os.path.isfile(("./data/online_PMNIST/0/train/set_" + str(set_num) + "/X.pt")) == False:
        print("No online dataset detected for PMNIST subsets. Creating new sets prior to loading task.")
        make_online_PMNIST()

    
    set_dict = {
    0:  {"dataset": "split_cifar", "task_num": 0, "set_num": 0},
    1:  {"dataset": "PMNIST", "task_num": 0, "set_num": 0},
    2:  {"dataset": "PMNIST", "task_num": 1, "set_num": 0},
    3:  {"dataset": "split_cifar", "task_num": 1, "set_num": 0},
    4:  {"dataset": "PMNIST", "task_num": 2, "set_num": 0},
    5:  {"dataset": "PMNIST", "task_num": 3, "set_num": 0},
    6:  {"dataset": "split_cifar", "task_num": 2, "set_num": 0},
    7:  {"dataset": "PMNIST", "task_num": 4, "set_num": 0},
    8:  {"dataset": "PMNIST", "task_num": 5, "set_num": 0},
    9:  {"dataset": "split_cifar", "task_num": 3, "set_num": 0},
    10: {"dataset": "PMNIST", "task_num": 0, "set_num": 1},
    11: {"dataset": "PMNIST", "task_num": 1, "set_num": 1},
    12: {"dataset": "split_cifar", "task_num": 4, "set_num": 0},
    13: {"dataset": "PMNIST", "task_num": 2, "set_num": 1},
    14: {"dataset": "PMNIST", "task_num": 3, "set_num": 1},
    15: {"dataset": "split_cifar", "task_num": 5, "set_num": 0},
    16: {"dataset": "PMNIST", "task_num": 4, "set_num": 1},
    17: {"dataset": "PMNIST", "task_num": 5, "set_num": 1},
    18: {"dataset": "split_cifar", "task_num": 6, "set_num": 0},
    19: {"dataset": "PMNIST", "task_num": 0, "set_num": 2},
    20: {"dataset": "PMNIST", "task_num": 1, "set_num": 2},
    21: {"dataset": "split_cifar", "task_num": 7, "set_num": 0},
    22: {"dataset": "PMNIST", "task_num": 2, "set_num": 2},
    23: {"dataset": "PMNIST", "task_num": 3, "set_num": 2},
    24: {"dataset": "split_cifar", "task_num": 8, "set_num": 0},
    25: {"dataset": "PMNIST", "task_num": 4, "set_num": 2},
    26: {"dataset": "PMNIST", "task_num": 5, "set_num": 2},
    27: {"dataset": "split_cifar", "task_num": 9, "set_num": 0}
    }

    assert set_num in set_dict.keys(), f"set_num {set_num} is not a valid set. Must be in range [0,27]"
    set_details = set_dict[set_num]
    dataset = set_details["dataset"]
    task_num = set_details["task_num"]
    task_set_num = set_details["set_num"]


    data={}

    if split == "train":
        pathx = ('../data/online_' + dataset + '/' + str(task_num) + "/train/" + str(task_set_num) + '/X.pt')
        pathy = ('../data/online_' + dataset + '/' + str(task_num) + "/train/" + str(task_set_num) + '/y.pt')
    
    elif split == "test":
        pathx = ('../data/online_' + dataset + '/' + str(task_num) + "/test/X.pt")
        pathy = ('../data/online_' + dataset + '/' + str(task_num) + "/test/y.pt")

    else:
        print("Invalid dataset split: ", split)
        raise ValueError    



    # print("Loading from: ",)
    # print("Path for X: ", pathx)
    # print("Path for Y: ", pathy)
    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)
    return data
    
    


### Not used in the experiments
### Frequently revisits PMNIST tasks 0 and 1, and only uses new tasks of CIFAR-100 splits
def get_online_simplemixed_cifar_pmnist(set_num:int = 0, split:str = "train"):

    if os.path.isfile(("./data/online_split_cifar/0/train/set_" + str(set_num) + "/X.pt")) == False:
        print("No online dataset detected for cifar subsets. Creating new sets prior to loading task.")
        make_online_splitcifar()
        
    if os.path.isfile(("./data/online_PMNIST/0/train/set_" + str(set_num) + "/X.pt")) == False:
        print("No online dataset detected for PMNIST subsets. Creating new sets prior to loading task.")
        make_online_PMNIST()

    set_dict = {
    0:  {"dataset": "split_cifar", "task_num": 0, "set_num": 0},
    1:  {"dataset": "PMNIST", "task_num": 0, "set_num": 0},
    2:  {"dataset": "PMNIST", "task_num": 1, "set_num": 0},
    3:  {"dataset": "split_cifar", "task_num": 1, "set_num": 0},
    4:  {"dataset": "PMNIST", "task_num": 0, "set_num": 1},
    5:  {"dataset": "PMNIST", "task_num": 1, "set_num": 1},
    6:  {"dataset": "split_cifar", "task_num": 2, "set_num": 0},
    7:  {"dataset": "PMNIST", "task_num": 0, "set_num": 2},
    8:  {"dataset": "PMNIST", "task_num": 1, "set_num": 2},
    9:  {"dataset": "split_cifar", "task_num": 3, "set_num": 0},
    10: {"dataset": "PMNIST", "task_num": 0, "set_num": 3},
    11: {"dataset": "PMNIST", "task_num": 1, "set_num": 3},
    12: {"dataset": "split_cifar", "task_num": 4, "set_num": 0},
    13: {"dataset": "PMNIST", "task_num": 0, "set_num": 4},
    14: {"dataset": "PMNIST", "task_num": 1, "set_num": 4},
    15: {"dataset": "split_cifar", "task_num": 5, "set_num": 0}
    }

    assert set_num in set_dict.keys(), f"set_num {set_num} is not a valid set. Must be in range [0,27]"
    set_details = set_dict[set_num]
    dataset = set_details["dataset"]
    task_num = set_details["task_num"]
    task_set_num = set_details["set_num"]


    data={}

    if split == "train":
        pathx = ('../data/online_' + dataset + '/' + str(task_num) + "/train/" + str(task_set_num) + '/X.pt')
        pathy = ('../data/online_' + dataset + '/' + str(task_num) + "/train/" + str(task_set_num) + '/y.pt')
    
    elif split == "test":
        pathx = ('../data/online_' + dataset + '/' + str(task_num) + "/test/X.pt")
        pathy = ('../data/online_' + dataset + '/' + str(task_num) + "/test/y.pt")

    else:
        print("Invalid dataset split: ", split)
        raise ValueError    


    # # print("Loading from: ",)
    # print("Path for X: ", pathx)
    # print("Path for Y: ", pathy)
    data['x'] = torch.load(pathx)
    data['y'] = torch.load(pathy)
    return data
    
    




















#!# RotMNIST needs to be reworked, so commenting out for now until I do so



# ### OMCPR
# ### Switches between tasks but RotMNIST gradually shifts in 10-degree increments over subsequent sets in which it appears
# def get_online_cifar_rotmnist(set_num:int = 0, split:str = "train"):
#     """
#     Sequence: 
#     23 tasks where:
#         set_num = 0: CIFAR100-split-1, 
#         set_num = 1: PMNIST-0-set-0, 
#         set_num = 2: RotMNIST-source, 
#         set_num = 3: CIFAR100-split-2, 
#         set_num = 4: PMNIST-0-set-1
#         set_num = 5: RotMNIST-inter-0
#         set_num = 6: CIFAR100-split-3, 
#         set_num = 7: PMNIST-0-set-2, 
#         set_num = 8: RotMNIST-inter-1
#         set_num = 9: CIFAR100-split-4, 
#         set_num = 10: PMNIST-0-set-3, 
#         set_num = 11: RotMNIST-inter-2
#         set_num = 12: CIFAR100-split-5, 
#         set_num = 13: PMNIST-0-set-4, 
#         set_num = 14: RotMNIST-inter-3
#         set_num = 15: CIFAR100-split-6, 
#         set_num = 16: PMNIST-0-set-5 
#         set_num = 17: RotMNIST-inter-4
#         set_num = 18: CIFAR100-split-7, 
#         set_num = 19: PMNIST-0-set-6, 
#         set_num = 20: RotMNIST-inter-5
#         set_num = 21: CIFAR100-split-8, 
#         set_num = 22: PMNIST-0-set-7, 
#     """

#     data={}
#     print
#     if set_num % 3 == 0:
#         set_num = int(set_num / 3) + 1
#         pathx = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/X.pt'
#         pathy = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/y.pt'
#     elif set_num %3 == 1:
#         set_num = int(math.floor(set_num/3))
#         print("Getting PMNIST 0 set number: ", set_num)
#         pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
#         pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
#     elif set_num % 3 == 2:
#         set_num = int(math.floor(set_num/3))
#         if set_num == 0:
#             print("Getting RotMNIST Source")
#             pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
#             pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
#         else:
#             print("Getting RotMNIST Intermediate: ", set_num)
#             pathx = ('../data/RotMNIST/Online/' + str(setsize) + "/inter/" + str(set_num-1) + '/0/X.pt')
#             pathy = ('../data/RotMNIST/Online/' + str(setsize) + "/inter/" + str(set_num-1) + '/0/y.pt')
#     if printpath == True:
#             # print("Loading from: ")
#             print("Path for X: ", pathx)
#             print("Path for Y: ", pathy)

#     data['x'] = torch.load(pathx)
#     data['y'] = torch.load(pathy)

#     # data['x'] = data['x'].type(torch.FloatTensor)
#     return data
    
    



# ### OMCPR-20
# ### Same as online_cifar_rotmnist but it skips in 20-degree increments of rotation for more drastic distribution shifts of RotMNIST
# def get_online_cifar_jump20rotmnist(set_num:int = 0, split:str = "train"):
#     """
#     Sequence: 
#     20 tasks where:
#         set_num = 0: CIFAR100-split-1, 
#         set_num = 1: PMNIST-0-set-0, 
#         set_num = 2: RotMNIST-source, 

#         set_num = 3: CIFAR100-split-2, 
#         set_num = 4: PMNIST-0-set-1

#         set_num = 5: CIFAR100-split-3, 
#         set_num = 6: PMNIST-0-set-2, 
#         set_num = 7: RotMNIST-inter-1

#         set_num = 8: CIFAR100-split-4, 
#         set_num = 9: PMNIST-0-set-3, 

#         set_num = 10: CIFAR100-split-5, 
#         set_num = 11: PMNIST-0-set-4, 
#         set_num = 12: RotMNIST-inter-3

#         set_num = 13: CIFAR100-split-6, 
#         set_num = 14: PMNIST-0-set-5 

#         set_num = 15: CIFAR100-split-7, 
#         set_num = 16: PMNIST-0-set-6, 
#         set_num = 17: RotMNIST-inter-5

#         set_num = 18: CIFAR100-split-8, 
#         set_num = 19: PMNIST-0-set-7, 
#     """

#     data={}
#     if set_num in [0,3,5,8,10,13,15,18]:
#         set_num = [0,3,5,8,10,13,15,18].index(set_num)+1
#         print("CIFAR Tasknum: ", set_num)
#         # set_num = int(set_num / 3) + 1
#         pathx = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/X.pt'
#         pathy = '../data/split_cifar/Online/'+ str(setsize) + "/" + str(set_num) + '/0/y.pt'
#     elif set_num in [1,4,6,9,11,14,16,19]:
#         set_num = [1,4,6,9,11,14,16,19].index(set_num)
#         # set_num = int(math.floor(set_num/3))
#         print("Getting PMNIST 0 set number: ", set_num)
#         pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
#         pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
#     elif set_num == 2:
#         print("Getting RotMNIST Source")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
#     elif set_num == 7:
#         print("Getting RotMNIST Intermediate: 1")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/1/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/1/0/y.pt')
#     elif set_num == 12:
#         print("Getting RotMNIST Intermediate: 3")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/3/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/3/0/y.pt')
#     elif set_num == 17:
#         print("Getting RotMNIST Intermediate: 5")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/y.pt')
#     if printpath == True:
#             # print("Loading from: ")
#             print("Path for X: ", pathx)
#             print("Path for Y: ", pathy)

#     data['x'] = torch.load(pathx)
#     data['y'] = torch.load(pathy)

#     return data
    





# ### OMCPR-30
# ### Same as online_cifar_rotmnist but it skips in 30-degree increments of rotation for more drastic distribution shifts of RotMNIST
# def get_online_cifar_jump30rotmnist(set_num:int = 0, split:str = "train"):
#     """
#     Sequence: 
#     19 tasks where:
#         set_num = 0: CIFAR100-split-1, 
#         set_num = 1: PMNIST-0-set-0, 
#         set_num = 2: RotMNIST-source,
        
#         set_num = 3: CIFAR100-split-2, 
#         set_num = 4: PMNIST-0-set-1
        
#         set_num = 5: CIFAR100-split-3, 
#         set_num = 6: PMNIST-0-set-2, 
        
#         set_num = 7: CIFAR100-split-4, 
#         set_num = 8: PMNIST-0-set-3, 
#         set_num = 9: RotMNIST-inter-2
        
#         set_num = 10: CIFAR100-split-5, 
#         set_num = 11: PMNIST-0-set-4, 
        
#         set_num = 12: CIFAR100-split-6, 
#         set_num = 13: PMNIST-0-set-5 
        
#         set_num = 14: CIFAR100-split-7, 
#         set_num = 15: PMNIST-0-set-6, 
#         set_num = 16: RotMNIST-inter-5
        
#         set_num = 17: CIFAR100-split-8, 
#         set_num = 18: PMNIST-0-set-7, 
#     """

#     data={}
#     if set_num in [0,3,5,7,10,12,14,17]:
#         set_num = [0,3,5,7,10,12,14,17].index(set_num)+1
#         print("CIFAR Tasknum: ", set_num)
#         pathx = '../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/X.pt'
#         pathy = '../data/split_cifar/Online/' + str(setsize) + "/" + str(set_num) + '/0/y.pt'
#     elif set_num in [1,4,6,8,11,13,15,18]:
#         set_num = [1,4,6,8,11,13,15,18].index(set_num)
#         print("PMNIST Tasknum: ", set_num)
#         print("Getting PMNIST 0 set number: ", set_num)
#         pathx = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/X.pt"
#         pathy = '../data/PMNIST/Online/' + str(setsize) + "/0/" + str(set_num) + "/y.pt"
#     elif set_num == 2:
#         set_num = int(math.floor(set_num/3))
#         print("Getting RotMNIST Source")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/source/0/y.pt')
#     elif set_num == 9:
#         print("Getting RotMNIST Intermediate: 2")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/2/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/2/0/y.pt')
#     elif set_num == 16:
#         print("Getting RotMNIST Intermediate: 5")
#         pathx = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/X.pt')
#         pathy = ('../data/RotMNIST/Online/' + str(setsize) + '/inter/5/0/y.pt')
#     if printpath == True:
#             # print("Loading from: ")
#             print("Path for X: ", pathx)
#             print("Path for Y: ", pathy)

#     data['x'] = torch.load(pathx)
#     data['y'] = torch.load(pathy)

#     return data
    






















#####################################################################
### Dataset creation functions
#####################################################################




def make_online_splitcifar(num_sets:int = 5, set_size:int = 1000):
    torch.manual_seed(0)
    data={}
    taskcla=[]
    size=[3,32,32]

    # CIFAR100
    dat={}

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    dat['train']=datasets.CIFAR100('../data/',train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.CIFAR100('../data/',train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

    for n in range(0,10):
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']=10
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}

    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        for image,target in loader:
            task_idx = target // 10
            data[task_idx.item()][s]['x'].append(image)
            data[task_idx.item()][s]['y'].append(target % 10)


    rootpath = '../data/online_split_cifar/'
    os.makedirs(rootpath, exist_ok=True)
    for t in range(0,10):
        for s in ['train','test']:
            data[t][s]['x'] = torch.cat(data[t][s]['x'], dim=0)
            data[t][s]['y'] = torch.cat(data[t][s]['y'], dim=0).to(torch.long)

            #!# Shuffle just to be careful. Make sure manual seed is set to 0 for consistency
            perm = torch.randperm(data[t][s]['x'].size(0))
            data[t][s]['x'] = data[t][s]['x'][perm]
            data[t][s]['y'] = data[t][s]['y'][perm]
        
        ### Save the test set for offline evaluation
        os.makedirs((rootpath + str(t) + "/test/") ,exist_ok=True)
        torch.save(data[t][s]['x'], (rootpath + str(t) + '/test/X.pt'))
        torch.save(data[t][s]['y'], (rootpath + str(t) + '/test/y.pt'))




        ### Split the training data into Z sets for online training 
        ### Note: We don't use all resulting sets, often only the first 1-3 for a given task so we cap it to saving num_sets=5 sets per task
        for i in range(min(num_sets,math.ceil(data[t]['train']['x'].size(0)/set_size))):
            Xsplit = data[t]['train']['x'][i*set_size:(i+1)*set_size]
            Ysplit = data[t]['train']['y'][i*set_size:(i+1)*set_size]
            
            savepath = rootpath + str(t) + "/train/set_" + str(i)
            os.makedirs(savepath, exist_ok =True)
            torch.save(Xsplit,(savepath+"/X.pt"))
            torch.save(Ysplit,(savepath+"/y.pt"))











def make_online_PMNIST(num_sets:int = 5, set_size:int = 1000):
    
    mnist_train = datasets.MNIST('../data/', train = True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        
    mnist_test = datasets.MNIST('../data/', train = False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),transforms.Resize((32,32))]), download = True)        

    dat={}
    data={}
    taskcla=[]
    size=[1,32,32]    
    os.makedirs('../data/online_PMNIST', exist_ok =True)
    
    dat['train']=mnist_train
    dat['test']=mnist_test
    
    ### Prepare the data variable and lists of label indices for further processing
    for t in range(0,6):
      data[t]={}
      data[t]['name']='PMNIST'
      data[t]['ncla']=10
      data[t]['train']={'x': [],'y': []}
      data[t]['test']={'x': [],'y': []}



    for t in range(0,6):
        torch.manual_seed(t)
        ### Permuted MNIST uses random per-task permutations of the image pixels to derive new versions of MNIST
        taskperm = torch.randperm((32*32))

        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            ### For each image we will flatten it, permute it according to taskperm, and then reshape it and convert it to produce (3,32,32) image shape
            for image,target in loader:   

                ### Flatten the (1,32,32) image into (1,1024)
                image = torch.flatten(image)
                image = image[taskperm].view(1,32,32)
                ### Gives shape (3,32,32)
                image = torch.cat((image,image,image), dim=0)

                data[t][s]['x'].append(image.unsqueeze(0))
                data[t][s]['y'].append(target)

            data[t][s]['x']=torch.cat(data[t][s]['x'], dim=0)
            data[t][s]['y']=torch.cat(data[t][s]['y'], dim=0).to(torch.long)
            print("Shape of x: ", data[t][s]['x'].shape)

            #!# Shuffle just to be careful. Make sure manual seed is set to 0 for consistency
            perm = torch.randperm(data[t][s]['x'].size(0))
            data[t][s]['x'] = data[t][s]['x'][perm]
            data[t][s]['y'] = data[t][s]['y'][perm]

    for t in range(0,6):
        ### Save test data for offline evaluation
        os.makedirs(('../data/online_PMNIST/' + str(t) + "/test/") ,exist_ok=True)
        torch.save(data[t]['test']['x'], ('../data/online_PMNIST/'+ str(t) + '/test/X.pt'))
        torch.save(data[t]['test']['y'], ('../data/online_PMNIST/'+ str(t) + '/test/y.pt'))
  
        path = "../data/online_PMNIST/" + str(t) + "/train/"

        ### Split the training data into Z sets for online training 
        ### Note: We don't use all resulting sets, only the first 1-3 for a given task so we cap it to saving num_sets=5 sets per task
        for i in range(min(num_sets,math.ceil(data[t]['train']['x'].size(0)/set_size))):
            Xsplit = data[t]['train']['x'][i*set_size:(i+1)*set_size]
            Ysplit = data[t]['train']['y'][i*set_size:(i+1)*set_size]
            
            savepath = "../data/online_PMNIST/" + str(t) + "/train/set_" + str(i)
            os.makedirs(savepath, exist_ok =True)
            torch.save(Xsplit,(savepath+"/X.pt"))
            torch.save(Ysplit,(savepath+"/y.pt"))
            
            
            