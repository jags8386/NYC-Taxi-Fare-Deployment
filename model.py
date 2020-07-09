import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

i = 0
for chunk in pd.read_csv('train.csv', chunksize=200000, nrows=1000000,
                         low_memory=False, parse_dates=['pickup_datetime']):
    if i == 0:
        df = chunk
        i += 1
        print("Read Chunk ----> ", i)
    else:
        df = pd.concat([df, chunk], axis=0)
        i += 1
        print("Read Chunk ----> ", i)


df = df[df.fare_amount >= 0]
df = df.dropna()


def select_within_bounding_box(df, bb):
    return (df.pickup_longitude >= bb[0]) & (df.pickup_longitude <= bb[1]) & \
           (df.pickup_latitude >= bb[2]) & (df.pickup_latitude <= bb[3]) & \
           (df.dropoff_longitude >= bb[0]) & (df.dropoff_longitude <= bb[1]) & \
           (df.dropoff_latitude >= bb[2]) & (df.dropoff_latitude <= bb[3])


BB = (-74.5, -72.8, 40.5, 41.8)
df2 = df[select_within_bounding_box(df, BB)]


# translate longitude/latitude coordinate into image xy coordinate
def lonlat_to_xy(longitude, latitude, dx, dy, bb):
    return (dx * (longitude - bb[0]) / (bb[1] - bb[0])).astype('int'), \
           (dy - dy * (latitude - bb[2]) / (bb[3] - bb[2])).astype('int')


nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:, :, 0] > 0.9

pickup_x, pickup_y = lonlat_to_xy(df2.pickup_longitude, df2.pickup_latitude,
                                  nyc_mask.shape[1], nyc_mask.shape[0], BB)
dropoff_x, dropoff_y = lonlat_to_xy(df2.dropoff_longitude, df2.dropoff_latitude,
                                    nyc_mask.shape[1], nyc_mask.shape[0], BB)

idx = (nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x])


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, bb):
        return (dx * (longitude - bb[0]) / (bb[1] - bb[0])).astype('int'), \
               (dy - dy * (latitude - bb[2]) / (bb[3] - bb[2])).astype('int')

    # define bounding box
    BB = (-74.5, -72.8, 40.5, 41.8)

    # read nyc mask and turn into boolean map with
    # land = True, water = False
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:, :, 0] > 0.9

    # calculate for each lon,lat coordinate the xy coordinate in the mask map
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude,
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude,
                                        nyc_mask.shape[1], nyc_mask.shape[0], BB)
    # calculate boolean index
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    # return only datapoints on land
    return df[idx]


df2 = remove_datapoints_from_water(df2)


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) *\
        (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


df2['distance_miles'] = distance(df2.pickup_latitude, df2.pickup_longitude, df2.dropoff_latitude,
                                 df2.dropoff_longitude)
df2 = df2[(df2.distance_miles >= 0.05)]
df2['year'] = df2.pickup_datetime.apply(lambda t: t.year)
df2['weekday'] = df2.pickup_datetime.apply(lambda t: t.weekday())
df2['hour'] = df2.pickup_datetime.apply(lambda t: t.hour)
df2['manhattans_distance'] = np.abs(df2['pickup_longitude'] - df2['dropoff_longitude']) + \
                             np.abs(df2['pickup_latitude'] - df2['dropoff_latitude'])

print("Generating Clean File ")

df_clean = df2[['fare_amount', 'passenger_count', 'weekday', 'hour', 'distance_miles',
                'manhattans_distance', 'year']]

df_train = df_clean[df_clean['year'] <= 2013].drop('year', axis=1)
df_test = df_clean[df_clean['year'] > 2013].drop('year', axis=1)

X_train = df_train.drop('fare_amount', axis=1)
X_test = df_test.drop('fare_amount', axis=1)
y_train = df_train['fare_amount']
y_test = df_test['fare_amount']

print("Running Linear Regresson ")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)

mse_dt = mse(y_test, lr_predict)

# Compute rmse_dt
rmse_dt = mse_dt**0.5

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

print("Dumping Model ")
# Creating a pickle file for the classifier
filename = 'NYC_Fare.pkl'
pickle.dump(lr, open(filename, 'wb'))
