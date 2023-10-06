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
path_df = os.path.join(path_dati,'traj_correct2.pkl')
df = pd.read_pickle(path_df) 


#creo un dataset ptrail formato dalle mov

dfp2 = PTRAILDataFrame(data_set = df, 
                      latitude = "lat", 
                      longitude = "lng", 
                      datetime = "datetime", 
                      traj_id = "uid",
                      rest_of_columns = ["alt", "label", "stop_id"])



#### evaluate the kinematic features
dfp2 = kinematic.generate_kinematic_features(dfp2)

# drop the rate of bearing_rate
dfp2 = dfp2.drop(["Rate_of_bearing_rate"], axis = 1)


dfp2 = dfp2.fillna(0)

#####################################
## filtering on speed and acceleration
#####################################

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

# rimuovo traiettorie con meno di 20 GPS points
dfp2 = filters.remove_trajectories_with_less_points(dfp2, num_min_points = 20)

dfp2 = filters.filter_by_bounding_box(dfp2, bounding_box = [-90, -180, 90, 180])


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

path_df_mov = os.path.join(path_dati, 'geolife_stop_mov_features_correct.pkl')
dfmov.to_pickle(path_df_mov)  # saves df to 'geolife_mov_with_features.pkl'