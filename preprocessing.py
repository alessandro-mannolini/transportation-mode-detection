# Preprocessing

import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import ptrail
# Use Ptrail 
from ptrail.core.TrajectoryDF import PTRAILDataFrame
from ptrail.features.kinematic_features import KinematicFeatures as kinematic
from ptrail.features.temporal_features import TemporalFeatures as temporal
from ptrail.preprocessing.statistics import Statistics as stat
from ptrail.preprocessing.filters import Filters as filters

import skmob
from skmob.preprocessing import detection, clustering, filtering
import os

# carico il file originale del dataset geolife in formato .pkl
path_dati = "C:\\Users\\aless\\Documents\\03_git_hub_personale\\03_transportation_mode_detection\\dati"
path_geolife = os.path.join(path_dati, "geolife.pkl")
df = pd.read_pickle(path_geolife) 

### remove the GPS points without labels
mask_label = df.label != 0
dfm = df[mask_label]

label_name = {1: "walk",
              2: "bike",
              3: "bus",
              4: "car",
              5: "subway",
              6: "train",
              7: "airplane",
              8: "boat",
              9: "run",
             10: "motorcycle",
             11: "taxi"}


dfm = dfm[(dfm.label!=7) & (dfm.label!=8) & (dfm.label!=9) & (dfm.label!=10)]        #### remove the labels: airplane, boat, run, motorway

dfm.label.mask(dfm.label == 11, 4, inplace = True)       ### put togheter taxi and car
dfm.label.mask(dfm.label == 6, 5, inplace = True)        ### put togheter subway and train

dfm.drop_duplicates(inplace = True)                     # Remove duplicated rows (keeping the first occurrence of each duplicates)

#dfm.to_pickle('geolife_label.pkl')  # saves df to 'geolife_label.pkl'


######################### create a TrajDataFrame from a pandas DataFrame ###########################

data_df = pd.DataFrame(dfm, columns=['user', 'lat', 'lon', 'alt','time','label'])
tdf = skmob.TrajDataFrame(data_df, latitude='lat', longitude = "lon", datetime='time', user_id='user')

# eliminate some outilers
ftdf = filtering.filter(tdf, max_speed_kmh=1000.0)

# create a list of stop
stdf = detection.stay_locations(ftdf, stop_radius_factor=0.5, 
                                minutes_for_a_stop=20.0, 
                                spatial_radius_km=0.2, 
                                leaving_time=True)

## creo un etichetta per ogni riga di ogni uid, al cambio uid riparte a contare 
stdf['stop_id'] = stdf.groupby('uid', as_index = False).cumcount()


# Adesso voglio effettuare dei merge/join
# primo Merge: uid, start_datetime e stop_id
# secondo Merge: uid, start_datetime e stop_id sul dataFrame del merge precedente

# rinonimo colonna datetime delle stops per effettuare un merge
stdf = stdf.rename(columns = {'datetime':'start_datetime'})

# riordino i due datasets
ftdf.sort_values(["uid", "datetime"], ascending = [True, True], inplace = True)
stdf.sort_values(["uid", "start_datetime"], ascending = [True, True], inplace = True)

#eseguo due merge, questo mi serve per avere inizio e fine delle stops
ftdf_start = pd.merge(ftdf, stdf[['uid', 'start_datetime', 'stop_id']], left_on = ['uid', "datetime"], right_on = ['uid', 'start_datetime'], how = "left")
start_end = pd.merge(ftdf_start, stdf[['uid', 'leaving_datetime', 'stop_id']], left_on= ['uid', 'datetime'], right_on=['uid', 'leaving_datetime'], how = 'left')

# completo i merge caricando i tempi di inizio e fine in avanti (forward fill)
# completo i merge caricando gli id in avanti (forward fill)

start_end.start_datetime.fillna(method = 'ffill', inplace = True)
start_end.leaving_datetime.fillna(method = 'ffill', inplace = True)

start_end.stop_id_x.fillna(method = 'ffill', inplace = True) 
start_end.stop_id_y.fillna(method = 'ffill', inplace = True)

#fill con -1 i nan che rimangono
start_end.stop_id_x.fillna(-1, inplace = True)
start_end.stop_id_y.fillna(-1, inplace = True)

#creo colonna con stop_id
start_end['stop_id'] = start_end['stop_id_x'] - start_end['stop_id_y']

#ci sono dei valori anomali, che rappresentano una differenza fra le due colonne stop_id_x e stop_id_y, quindi sono delle stop e le rinomino con 1
start_end.loc[start_end['stop_id']!=0, 'stop_id'] = 1


### elimino le colonne che non mi servono e salvo il dataset in formato .pkl

start_end.drop(['start_datetime', 'leaving_datetime', 'stop_id_x', 'stop_id_y'], axis = 1, inplace = True)
###############################################################################################################
### Salvo il dataframe start_end 
### in modo da poterlo usare per effettuare il preprocessing
### nel file preprocessing_stop_mov.py
################################################################################################################

path_start_end = os.path.join(path_dati, 'traj_correct2.pkl')
start_end.to_pickle(path_start_end)      



### Filtro il dataset fra stop e mov e creo due datasets, uno solo per le mov e uno solo per le stop

## creo un dataframe delle sole mov
mov = pd.DataFrame(start_end[start_end["stop_id"] == 0])
mov.drop('stop_id', axis = 1, inplace = True)
print('mov shape: ', mov.shape)

## creo un dataframe delle sole stops
stop = pd.DataFrame(start_end[start_end['stop_id']== 1])
stop.drop("stop_id", axis = 1, inplace = True)
print('stop shape: ', stop.shape)

## salvo i due dataset ottenuti
path_mov = os.path.join(path_dati, 'mov_correct2.pkl')
mov.to_pickle(path_mov)

path_stop = os.path.join(path_dati, 'stop_correct2.pkl')
stop.to_pickle(path_stop)


#creo un dataset ptrail formato dalle mov
dfp2 = PTRAILDataFrame(data_set = mov, 
                      latitude = "lat", 
                      longitude = "lng", 
                      datetime = "datetime", 
                      traj_id = "uid",
                      rest_of_columns = ["alt", "label"])

#### evaluate the kinematic features
dfp2 = kinematic.generate_kinematic_features(dfp2)

# drop the rate of bearing_rate
dfp2 = dfp2.drop(["Rate_of_bearing_rate"], axis = 1)
dfp2 = dfp2.fillna(0)

########################################
## filtering on speed and acceleration #
########################################

# m/s

max_speed_walk = 7
max_speed_bike = 12
max_speed_bus = 34
max_speed_car = 50
max_speed_train = 34

# m/s^2

max_acc_walk = 3
max_acc_bike = 3
max_acc_bus = 2
max_acc_car = 10
max_acc_train = 3

label_name_new = {1: "walk",
                  2: "bike",
                  3: "bus",
                  4: "car&taxi",
                  5: "subway&train"}


index_walk = dfp2[(dfp2.label == 1) & ((dfp2.Speed > max_speed_walk) | (dfp2.Acceleration > max_acc_walk))].index
dfp2.drop(index_walk, inplace = True)

index_bike = dfp2[(dfp2.label == 2) & ((dfp2.Speed > max_speed_bike) | (dfp2.Acceleration > max_acc_bike))].index
dfp2.drop(index_bike, inplace = True)

index_bus = dfp2[(dfp2.label == 3) & ((dfp2.Speed > max_speed_bus) | (dfp2.Acceleration > max_acc_bus))].index
dfp2.drop(index_bus, inplace = True)

index_car = dfp2[(dfp2.label == 4) & ((dfp2.Speed > max_speed_car) | (dfp2.Acceleration > max_acc_car))].index
dfp2.drop(index_car, inplace = True)

index_train = dfp2[(dfp2.label == 5) & ((dfp2.Speed > max_speed_train) | (dfp2.Acceleration > max_acc_train))].index
dfp2.drop(index_train, inplace = True)

dfp2 = filters.remove_trajectories_with_less_points(dfp2, num_min_points = 20)              # rimuovo traiettorie con meno di 20 GPS points
dfp2 = filters.filter_by_bounding_box(dfp2, bounding_box = [-90, -180, 90, 180])            # Filtro sulle coordinate geografiche


#### generate temporal features

# create time column
dfp2 = temporal.create_time_column(dfp2)

# create day of the week column 
dfp2 = temporal.create_day_of_week_column(dfp2)

# create time of day column
dfp2 = temporal.create_time_of_day_column(dfp2)

### create the mov_id column
dfmov = dfp2.reset_index()

dfmov['valori_combinati'] = dfmov['traj_id'].astype(str) + '-' + dfmov['label'].astype(str)
dfmov['cambio_valore'] = (dfmov['valori_combinati'] != dfmov['valori_combinati'].shift(1)).astype(int)

dfmov['mov_id'] = dfmov['cambio_valore'].cumsum().astype(str)
dfmov[['traj_id', 'label']] = dfmov['valori_combinati'].str.split('-', expand=True)
dfmov = dfmov.drop(columns=['valori_combinati'])

path_dfmov = os.path.join(path_dati, "geolife_mov_with_features_correct2.pkl")
dfmov.to_pickle(path_dfmov)  # saves df to 'geolife_mov_with_features.pkl'