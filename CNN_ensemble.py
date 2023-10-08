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
from functions import train_cnn_exp

#####################################################################################################
# parameters
with open("config_ensemble.yaml", "r") as f:
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

# standard scaler 
scaler = StandardScaler()
features_scaler = traj1[['Speed','Acceleration','Jerk','Distance','Distance_from_start','alt','Bearing_Rate','Bearing']]
scaled_data = scaler.fit_transform(features_scaler)
traj1[['Speed','Acceleration','Jerk','Distance','Distance_from_start','alt','Bearing_Rate','Bearing']] = scaled_data

################################################################################################################################

x_channels02 = traj1.groupby(['traj_id','mov_id'], as_index = False).apply(lambda x: x[features].values)
x_channels02 = x_channels02.values

x_cut = [x[:900] for x in x_channels02]
max_len = 900   # ho tagliato le traiettorie a 900
x_cut_0 = [np.pad(x, ((0, max_len - len(x)), (0,0)) , pad)  for x in x_cut]
x_cut_0_array = np.array(x_cut_0)

# labels
y = df1.groupby(['traj_id','mov_id', "label"]).apply(lambda x: x['label'].values)
y = np.array(y.index.get_level_values("label")).astype(int)
y2 = y-1


# splitto X e y in train e test set
x_trainval, x_test, y_trainval, y_test = train_test_split(x_cut_0_array, y2, test_size=0.25, random_state=77)
y_trainval, y_test = y_trainval.astype(float), y_test.astype(float)


# 1
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 11)
# 2
x_train2, x_val2, y_train2, y_val2 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 22)
# 3
x_train3, x_val3, y_train3, y_val3 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 33)
# 4
x_train4, x_val4, y_train4, y_val4 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 44)
# 5
x_train5, x_val5, y_train5, y_val5 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 55)
# 6
x_train6, x_val6, y_train6, y_val6 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 66)
# 7
x_train7, x_val7, y_train7, y_val7 = train_test_split(x_trainval, y_trainval, test_size = 0.25, random_state = 77)


# test
x_test = torch.FloatTensor(x_test)

# train
x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7 = map(torch.FloatTensor, (x_train1,
                                                                                               x_train2,
                                                                                               x_train3,
                                                                                               x_train4,
                                                                                               x_train5,
                                                                                               x_train6,
                                                                                               x_train7))
# val
x_val1, x_val2, x_val3, x_val4, x_val5, x_val6, x_val7 = map(torch.FloatTensor, (x_val1,
                                                                                 x_val2,
                                                                                 x_val3,
                                                                                 x_val4,
                                                                                 x_val5,
                                                                                 x_val6,
                                                                                 x_val7))


# test
x_test_tensor = x_test.permute((0, 2, 1))

# train
x_train1_tensor = x_train1.permute((0, 2, 1))
x_train2_tensor = x_train2.permute((0, 2, 1))
x_train3_tensor = x_train3.permute((0, 2, 1))
x_train4_tensor = x_train4.permute((0, 2, 1))
x_train5_tensor = x_train5.permute((0, 2, 1))
x_train6_tensor = x_train6.permute((0, 2, 1))
x_train7_tensor = x_train7.permute((0, 2, 1))

# val
x_val1_tensor = x_val1.permute((0, 2, 1))
x_val2_tensor = x_val2.permute((0, 2, 1))
x_val3_tensor = x_val3.permute((0, 2, 1))
x_val4_tensor = x_val4.permute((0, 2, 1))
x_val5_tensor = x_val5.permute((0, 2, 1))
x_val6_tensor = x_val6.permute((0, 2, 1))
x_val7_tensor = x_val7.permute((0, 2, 1))

# modifico la dimensione dei tensori

# test
x_test_tensor_reshaped = x_test_tensor.reshape([-1,channels,1,900])

# train
x_train1_tensor_reshaped = x_train1_tensor.reshape([-1,channels,1,900])
x_train2_tensor_reshaped = x_train2_tensor.reshape([-1,channels,1,900])
x_train3_tensor_reshaped = x_train3_tensor.reshape([-1,channels,1,900])
x_train4_tensor_reshaped = x_train4_tensor.reshape([-1,channels,1,900])
x_train5_tensor_reshaped = x_train5_tensor.reshape([-1,channels,1,900])
x_train6_tensor_reshaped = x_train6_tensor.reshape([-1,channels,1,900])
x_train7_tensor_reshaped = x_train7_tensor.reshape([-1,channels,1,900])

# val 
x_val1_tensor_reshaped = x_val1_tensor.reshape([-1, channels, 1, 900])
x_val2_tensor_reshaped = x_val2_tensor.reshape([-1, channels, 1, 900])
x_val3_tensor_reshaped = x_val3_tensor.reshape([-1, channels, 1, 900])
x_val4_tensor_reshaped = x_val4_tensor.reshape([-1, channels, 1, 900])
x_val5_tensor_reshaped = x_val5_tensor.reshape([-1, channels, 1, 900])
x_val6_tensor_reshaped = x_val6_tensor.reshape([-1, channels, 1, 900])
x_val7_tensor_reshaped = x_val7_tensor.reshape([-1, channels, 1, 900])

# sistemo anche le labels come tensori

y_test_tensor = torch.LongTensor(y_test)

# train
y_train1_tensor, y_train2_tensor,y_train3_tensor, y_train4_tensor, y_train5_tensor, y_train6_tensor, y_train7_tensor = map(
                      torch.LongTensor, (y_train1,
                                         y_train2,
                                         y_train3,
                                         y_train4,
                                         y_train5,
                                         y_train6,
                                         y_train7))

# val
y_val1_tensor, y_val2_tensor, y_val3_tensor, y_val4_tensor, y_val5_tensor, y_val6_tensor, y_val7_tensor = map(
                        torch.LongTensor, (y_val1,
                                           y_val2,
                                           y_val3, 
                                           y_val4,
                                           y_val5,
                                           y_val6,
                                           y_val7))


batch_size = batch

# test
test_ds = TensorDataset(x_test_tensor_reshaped, y_test_tensor)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# train
train1_ds = TensorDataset(x_train1_tensor_reshaped, y_train1_tensor)
train1_dl = DataLoader(train1_ds, batch_size=batch_size, shuffle = True)

train2_ds = TensorDataset(x_train2_tensor_reshaped, y_train2_tensor)
train2_dl = DataLoader(train2_ds, batch_size=batch_size, shuffle = True)

train3_ds = TensorDataset(x_train3_tensor_reshaped, y_train3_tensor)
train3_dl = DataLoader(train3_ds, batch_size=batch_size, shuffle = True)

train4_ds = TensorDataset(x_train4_tensor_reshaped, y_train4_tensor)
train4_dl = DataLoader(train4_ds, batch_size=batch_size, shuffle = True)

train5_ds = TensorDataset(x_train5_tensor_reshaped, y_train5_tensor)
train5_dl = DataLoader(train5_ds, batch_size=batch_size, shuffle = True)

train6_ds = TensorDataset(x_train6_tensor_reshaped, y_train6_tensor)
train6_dl = DataLoader(train6_ds, batch_size=batch_size, shuffle = True)

train7_ds = TensorDataset(x_train7_tensor_reshaped, y_train7_tensor)
train7_dl = DataLoader(train7_ds, batch_size=batch_size, shuffle = True)

# val
val1_ds = TensorDataset(x_val1_tensor_reshaped, y_val1_tensor)
val1_dl = DataLoader(val1_ds, batch_size=batch_size, shuffle = True)

val2_ds = TensorDataset(x_val2_tensor_reshaped, y_val2_tensor)
val2_dl = DataLoader(val2_ds, batch_size=batch_size, shuffle = True)

val3_ds = TensorDataset(x_val3_tensor_reshaped, y_val3_tensor)
val3_dl = DataLoader(val3_ds, batch_size=batch_size, shuffle = True)

val4_ds = TensorDataset(x_val4_tensor_reshaped, y_val4_tensor)
val4_dl = DataLoader(val4_ds, batch_size=batch_size, shuffle = True)

val5_ds = TensorDataset(x_val5_tensor_reshaped, y_val5_tensor)
val5_dl = DataLoader(val5_ds, batch_size=batch_size, shuffle = True)

val6_ds = TensorDataset(x_val6_tensor_reshaped, y_val6_tensor)
val6_dl = DataLoader(val6_ds, batch_size=batch_size, shuffle = True)

val7_ds = TensorDataset(x_val7_tensor_reshaped, y_val7_tensor)
val7_dl = DataLoader(val7_ds, batch_size=batch_size, shuffle = True)


###################################################################################################################################

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
    

#######################################################################################################################


# model 1
cnn1 = CNN()
optimizer = optim.Adam(cnn1.parameters(), lr=lr)
results1 = train_cnn_exp(cnn1, optimizer, torch.nn.CrossEntropyLoss(), train1_dl, val1_dl, epochs=epochs, to_print = False)

# Model 2
cnn2 = CNN()
optimizer = optim.Adam(cnn2.parameters(), lr=lr)
results2 = train_cnn_exp(cnn2, optimizer, torch.nn.CrossEntropyLoss(), train2_dl, val2_dl, epochs=epochs, to_print = False)

# Model 3
cnn3 = CNN()
optimizer = optim.Adam(cnn3.parameters(), lr=lr)
results3 = train_cnn_exp(cnn3, optimizer, torch.nn.CrossEntropyLoss(), train3_dl, val3_dl, epochs=epochs, to_print = False)

# Model 4
cnn4 = CNN()
optimizer = optim.Adam(cnn4.parameters(), lr=lr)
results4 = train_cnn_exp(cnn4, optimizer, torch.nn.CrossEntropyLoss(), train4_dl, val4_dl, epochs=epochs, to_print = False)

# Model 5
cnn5 = CNN()
optimizer = optim.Adam(cnn5.parameters(), lr=lr)
results5 = train_cnn_exp(cnn5, optimizer, torch.nn.CrossEntropyLoss(), train5_dl, val5_dl, epochs=epochs, to_print = False)

# Model 6
cnn6 = CNN()
optimizer = optim.Adam(cnn6.parameters(), lr=lr)
results6 = train_cnn_exp(cnn6, optimizer, torch.nn.CrossEntropyLoss(), train6_dl, val6_dl, epochs=epochs, to_print = False)

# Model 7
cnn7 = CNN()
optimizer = optim.Adam(cnn7.parameters(), lr=lr)
results7 = train_cnn_exp(cnn7, optimizer, torch.nn.CrossEntropyLoss(), train7_dl, val7_dl, epochs=epochs, to_print = False)

#####################################################################################################################################


# Salvo risultati
stringa_data = datetime.date.today().strftime("%Y_%m_%d")

name = stringa_data + '_CNN_ensemble_' + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj    # nome con cui salvare il file excel
name_excel = name + '.xlsx'
path_excel = os.path.join(folder_risultati, name_excel)

res = [results1, results2, results3, results4, results5, results6, results7]
sheet_names = ['results1', 'results2', 'results3', 'results4', 'results5', 'results6', 'results7']

for j in range(len(res)):
    if j == 0:
        res[j].to_excel(path_excel, sheet_name=sheet_names[j], index = False)
    else:
        with pd.ExcelWriter(path_excel, mode='a', engine='openpyxl') as writer:
            res[j].to_excel(writer, sheet_name=sheet_names[j], index = False)

# salvo i modelli nei percorsi (paths)
path1 = stringa_data + "_CNN_ensemble1_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path2 = stringa_data + "_CNN_ensemble2_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path3 = stringa_data + "_CNN_ensemble3_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path4 = stringa_data + "_CNN_ensemble4_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path5 = stringa_data + "_CNN_ensemble5_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path6 = stringa_data + "_CNN_ensemble6_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"
path7 = stringa_data + "_CNN_ensemble7_" + str(channels) + 'f_' + pad + '_' + str(epochs) +'_'+ name_traj + ".pth"

models_path = [os.path.join(folder_modelli, path1), 
               os.path.join(folder_modelli, path2), 
               os.path.join(folder_modelli, path3), 
               os.path.join(folder_modelli, path4), 
               os.path.join(folder_modelli, path5), 
               os.path.join(folder_modelli, path6), 
               os.path.join(folder_modelli, path7)]
models = [cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7]

for i in range(len(models_path)):
    torch.save(models[i].state_dict(), models_path[i])


# carico i modelli dai percorsi in cui sono stati salvati
models = []
for path in models_path:
    model = CNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    models.append(model)


### Valuto test set e calcolo le misure
y_test_pred = torch.zeros(32)
y_test_true = torch.zeros(32)

with torch.no_grad():
    num_correct = 0
    num_examples  = 0
    for X_batch, targets in test_dl:
        X_batch = X_batch.to(device)
        targets = targets.to(device)
        outputs = []
        for model in models:
            output = model(X_batch)
            output = torch.max(F.softmax(output, dim=1), dim=1)[1]
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=0)
        outputs, _ = torch.mode(outputs, dim=0)
        correct = torch.eq(outputs, targets)
        num_correct += torch.sum(correct).item()
        num_examples += correct.shape[0]
        acc = num_correct/num_examples
        y_test_pred = torch.cat((y_test_pred, outputs), 0)
        y_test_true = torch.cat((y_test_true, targets), 0)
        
y_test_pred = y_test_pred[batch:]
y_test_true = y_test_true[batch:]

print("The Accuracy on the test set is: {}".format(acc))

cf_test1 = confusion_matrix(y_test_true, y_test_pred) 
cf_test1 = pd.DataFrame(cf_test1, columns = [1,2,3,4,5], index = [1,2,3,4,5])
cf_test1["tot_label"] = cf_test1[[1,2,3,4,5]].sum(axis = 1)
cf_test1["accuracy_label"] = 0
cf_test1.loc[1,"accuracy_label"] = cf_test1.loc[1,1]/cf_test1.loc[1,"tot_label"]
cf_test1.loc[2,"accuracy_label"] = cf_test1.loc[2,2]/cf_test1.loc[2,"tot_label"]
cf_test1.loc[3,"accuracy_label"] = cf_test1.loc[3,3]/cf_test1.loc[3,"tot_label"]
cf_test1.loc[4,"accuracy_label"] = cf_test1.loc[4,4]/cf_test1.loc[4,"tot_label"]
cf_test1.loc[5,"accuracy_label"] = cf_test1.loc[5,5]/cf_test1.loc[5,"tot_label"]
accur = accuracy_score(y_test_true, y_test_pred)
f1 = f1_score(y_test_true, y_test_pred, average = 'macro')
cf_test1['accuracy'] = accur
cf_test1['f1_score'] = f1

with pd.ExcelWriter(path_excel, mode='a', engine='openpyxl') as writer:
    cf_test1.to_excel(writer, sheet_name="CNN Test")