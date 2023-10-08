import pandas as pd
import numpy as np
import time
import openpyxl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import yaml
import os
import datetime
from functions import train_cnn

#####################################################################################################
# parameters
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

flag = config['parameters']['flag']
epochs = config['parameters']['epochs']
batch = config['parameters']['batch_size']
lr = config['parameters']['learning_rate']
features = config['parameters']['features']
pad = config['parameters']['pad']

# folder
folder_dati = config['folders']['folder_dati']
folder_modelli = config['folders']['folder_modelli']
folder_risultati = config['folders']['folder_risultati']
######################################################################################################

# check della gpu
# device = cuba if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if flag:
    path_dati = os.path.join(folder_dati, "geolife_stop_mov_features_correct.pkl")
    df1 = pd.read_pickle(path_dati)
    name_traj = 'stop_mov'
    features = ['stop_id'].extend(features)
    channels = len(features)
else:
    path_dati = os.path.join(folder_dati, "geolife_mov_with_features_correct2.pkl")
    df1 = pd.read_pickle(path_dati)
    name_traj = 'mov'
    channels = len(features)

df1 = df1.groupby(["mov_id"]).filter(lambda x: len(x) >= 10) 
traj1 = df1.copy()

scaler = StandardScaler()
features_scaler = traj1[['Speed','Acceleration','Jerk','Distance','Distance_from_start','alt','Bearing_Rate','Bearing']]
scaled_data = scaler.fit_transform(features_scaler)
traj1[['Speed','Acceleration','Jerk','Distance','Distance_from_start','alt','Bearing_Rate','Bearing']] = scaled_data

#################################################################################################################################

#################################################################################################################################

x_channels = traj1.groupby(['traj_id','mov_id'], as_index = False).apply(lambda x: x[features].values)
x_channels = x_channels.values

x_cut = [x[:900] for x in x_channels]
max_len = 900   # ho tagliato le traiettorie a 900
x_cut_0 = [np.pad(x, ((0, max_len - len(x)), (0,0)) , pad)  for x in x_cut]
x_cut_0_array = np.array(x_cut_0)

# labels
y = df1.groupby(['traj_id','mov_id', "label"]).apply(lambda x: x['label'].values)
y = np.array(y.index.get_level_values("label")).astype(int)
y2 = y-1

# splitto X e y in train e test set
x_train, x_test, y_train, y_test = train_test_split(x_cut_0_array, y2, test_size=0.25, random_state=77)
y_train, y_test = y_train.astype(float), y_test.astype(float)

x_train, x_test= map(torch.FloatTensor,(x_train, x_test))

x_train_tensor = x_train.permute((0, 2, 1))
x_test_tensor = x_test.permute((0, 2, 1))

x_train_tensor_reshaped = x_train_tensor.reshape([-1,channels,1,900])
x_test_tensor_reshaped = x_test_tensor.reshape([-1,channels,1,900])

y_train_tensor, y_test_tensor = map(torch.LongTensor, (y_train, y_test))

batch_size = batch

train_ds = TensorDataset(x_train_tensor_reshaped, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle = True)

test_ds = TensorDataset(x_test_tensor_reshaped, y_test_tensor)
test_dl = DataLoader(test_ds, batch_size=batch_size)

##############################################################################################################################

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels = channels, out_channels=32,
                              kernel_size =(1,3) , stride=1 , padding= 0) 
        self.relu1 = nn.ReLU()
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 32,
                             kernel_size = (1,3), stride = 1, padding = 0)
        self.relu2 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2))

        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=(1,3), stride=1, padding=0)
        self.relu3 = nn.ReLU()
        
        # Convolution 4
        self.cnn4 = nn.Conv2d(in_channels = 64, out_channels = 64,
                             kernel_size = (1,3), stride = 1, padding = 0)
        self.relu4 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2))
        
        # Convolution 5
        
        self.cnn5 = nn.Conv2d(in_channels = 64, out_channels = 128,
                             kernel_size = (1,3), stride = 1, padding = 0)
        
        self.relu5 = nn.ReLU()
        
        # Convolution 6 
        
        self.cnn6 = nn.Conv2d(in_channels = 128, out_channels = 128,
                             kernel_size = (1,3), stride = 1, padding = 0)
        self.relu6 = nn.ReLU()
        
        # Max pool 3 
        self.maxpool3 = nn.MaxPool2d(kernel_size = (1,2))
        
        # Drop Out 1
        
        self.dropout1 = nn.Dropout2d(0.5)
        

        # Fully connected 1
        self.fc1 = nn.Linear(128 * 109 * 1, 32 * 109) #1/4
        
        self.relu7 = nn.ReLU()
        
        # Drop Out 2
        
        self.dropout2 = nn.Dropout(0.5)   #usare dropout 1 d
        
        # Fully connected 2
        
        self.fc2 = nn.Linear(32 * 109, 5)

    def forward(self, x, to_print=False):
        # Set 1
        if to_print:
            print('INPUT',x.shape)
        out = self.cnn1(x)
        if to_print:
            print('CNN1',out.shape)
        out = self.relu1(out)
        
        out = self.cnn2(out)
        if to_print:
            print("CNN2", out.shape)
        out = self.relu2(out)
                
        out = self.maxpool1(out)
        if to_print:
            print('MAXPOOL1',out.shape)
        
        # Set 2
        out = self.cnn3(out)
        if to_print:
            print('CNN3',out.shape)

        out = self.relu3(out)
        
        out = self.cnn4(out)
        
        if to_print:
            print('CNN4',out.shape)

        out = self.relu4(out)

        out = self.maxpool2(out)
        if to_print:
            print("after the 3rd maxpool:{} ".format(out.shape))
        
        out = self.dropout1(out)
        
        # set 3
        out = self.cnn5(out)
        if to_print:
            print('CNN5',out.shape)

        out = self.relu5(out)
        
        out = self.cnn6(out)
        
        if to_print:
            print('CNN6',out.shape)

        out = self.relu6(out)

        out = self.maxpool3(out)
        if to_print:
            print("after the 3rd maxpool:{} ".format(out.shape))
        
        out = self.dropout1(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        if to_print:
            print("after the flatten:{} ".format(out.shape))
        out = self.fc1(out)
        
        out = self.relu7(out)
        
        out = self.dropout2(out)
        
        out = out.view(out.size(0), -1)
        if to_print:
            print("after the flatten:{} ".format(out.shape))
        out = self.fc2(out)
        
        if to_print:
            print('FINAL',out.shape)

        return out

########################################################################################################

cnn = CNN()
optimizer = optim.Adam(cnn.parameters(), lr=lr)
results = train_cnn(cnn, optimizer, torch.nn.CrossEntropyLoss(), train_dl, test_dl, epochs=epochs, to_print = False)

# Salvo modello e risultati
stringa_data = datetime.date.today().strftime("%Y_%m_%d")

name = stringa_data + '_CNN_' + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj    # nome con cui salvare il file excel
name_excel = name + '.xlsx'
path_excel = os.path.join(folder_risultati, name_excel)
results.to_excel(path_excel, sheet_name = 'CNN results', index=False)

name_model = name + '.pth'
path_model = os.path.join(folder_modelli, name_model)
torch.save(cnn.state_dict(), path_model)     # salvo i parametri del modello
cnn.load_state_dict(torch.load(path_model))  # carico i parametri del modello


### valuto train set e calcolo le misure che voglio###
y_train_pred = torch.zeros(size = (batch,5))
y_train_true = torch.zeros(batch)

with torch.no_grad():
    cnn.eval()
    num_correct = 0
    num_examples  = 0
    for X_batch, targets in train_dl:
        X_batch = X_batch.to(device)
        targets = targets.to(device)
        #model
        y_train_pred1 = cnn(X_batch)
        correct = torch.eq(torch.max(F.softmax(y_train_pred1, dim=1), dim=1)[1], targets)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
        acc = num_correct/num_examples

        y_train_pred = torch.cat((y_train_pred, y_train_pred1), 0)
        y_train_true = torch.cat((y_train_true, targets), 0)
        
y_train_pred = y_train_pred[batch:]
y_train_true = y_train_true[batch:]

y_train_predicted1 = y_train_pred.argmax(1)
cf_train1 = confusion_matrix(y_train_true, y_train_predicted1) 
cf_train1 = pd.DataFrame(cf_train1, columns = [1,2,3,4,5], index = [1,2,3,4,5])
cf_train1["tot_label"] = cf_train1[[1,2,3,4,5]].sum(axis = 1)
cf_train1["accuracy_label"] = 0
cf_train1.loc[1,"accuracy_label"] = cf_train1.loc[1,1]/cf_train1.loc[1,"tot_label"]
cf_train1.loc[2,"accuracy_label"] = cf_train1.loc[2,2]/cf_train1.loc[2,"tot_label"]
cf_train1.loc[3,"accuracy_label"] = cf_train1.loc[3,3]/cf_train1.loc[3,"tot_label"]
cf_train1.loc[4,"accuracy_label"] = cf_train1.loc[4,4]/cf_train1.loc[4,"tot_label"]
cf_train1.loc[5,"accuracy_label"] = cf_train1.loc[5,5]/cf_train1.loc[5,"tot_label"]
accur = accuracy_score(y_train_true, y_train_predicted1)
f1 = f1_score(y_train_true, y_train_predicted1, average = 'macro')
cf_train1['accuracy'] = accur
cf_train1['f1_score'] = f1

with pd.ExcelWriter(path_excel, mode='a', engine='openpyxl') as writer:
    cf_train1.to_excel(writer, sheet_name="CNN Train")


### calcolo test set e le misure ###
y_test_pred = torch.zeros(size = (batch,5))
y_test_true = torch.zeros(batch)

with torch.no_grad():
    cnn.eval()
    num_correct = 0
    num_examples  = 0
    for X_batch, targets in test_dl:
        X_batch = X_batch.to(device)
        targets = targets.to(device)
        y_test_pred1 = cnn(X_batch)
        
        correct = torch.eq(torch.max(F.softmax(y_test_pred1, dim=1), dim=1)[1], targets)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
        acc = num_correct/num_examples
        y_test_pred = torch.cat((y_test_pred, y_test_pred1), 0)
        y_test_true = torch.cat((y_test_true, targets), 0)
        
y_test_pred = y_test_pred[batch:]
y_test_true = y_test_true[batch:]


y_test_predicted1 = y_test_pred.argmax(1)

cf_test1 = confusion_matrix(y_test_true, y_test_predicted1) 
cf_test1 = pd.DataFrame(cf_test1, columns = [1,2,3,4,5], index = [1,2,3,4,5])
cf_test1["tot_label"] = cf_test1[[1,2,3,4,5]].sum(axis = 1)
cf_test1["accuracy_label"] = 0
cf_test1.loc[1,"accuracy_label"] = cf_test1.loc[1,1]/cf_test1.loc[1,"tot_label"]
cf_test1.loc[2,"accuracy_label"] = cf_test1.loc[2,2]/cf_test1.loc[2,"tot_label"]
cf_test1.loc[3,"accuracy_label"] = cf_test1.loc[3,3]/cf_test1.loc[3,"tot_label"]
cf_test1.loc[4,"accuracy_label"] = cf_test1.loc[4,4]/cf_test1.loc[4,"tot_label"]
cf_test1.loc[5,"accuracy_label"] = cf_test1.loc[5,5]/cf_test1.loc[5,"tot_label"]
accur = accuracy_score(y_test_true, y_test_predicted1)
f1 = f1_score(y_test_true, y_test_predicted1, average = 'macro')
cf_test1['accuracy'] = accur
cf_test1['f1_score'] = f1
with pd.ExcelWriter(path_excel, mode='a', engine='openpyxl') as writer:
    cf_test1.to_excel(writer, sheet_name="CNN Test")