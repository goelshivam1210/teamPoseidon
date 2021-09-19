# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller,acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pylab import rcParams


# %%
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
df=pd.read_csv('final_dataset_csv.csv', sep=',',header=0)
df = df.ffill()
test_data_size = (int(df.shape[0] * 0.20))
df.drop(df.tail(test_data_size).index, inplace = True) # remove and set aside 20% of dataset for testing.

TRAIN_SPLIT = int(df.shape[0]*0.70) # 80% training 20 % validation
# they definitely needs optimization.
BATCH_SIZE = 32 # bacth size in batch-SGD/variants
BUFFER_SIZE = 64 # for shuffling the dataset
STEP = 1 # for creation of dataset

# Train and evaluate
STEPS_PER_EPOCH = 50 # hyperparameter
EPOCHS = 20 # hyperparameter


# %%
# x = df.drop('RES_LEVEL_FT', axis=1)
# y = df['RES_LEVEL_FT']
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle = False)
# x.columns


# %%



# %%
# univariate data
uni_data_df = df['RES_LEVEL_FT']
uni_data_df.index = df['FLOW_DATE']
uni_data_df.head()
uni_data_df.plot()


# %%
df.head()
df.columns
df[['OUTFLOW_CUECS']]
features = df[['PRESENT_STORAGE_TMC', 'INFLOW_CUSECS', 'OUTFLOW_CUECS', 'tempC', 'windspeedKmph', 'precipMM', 'humidity', 'pressure (mB)', 'cloudcover (%)', 'HeatIndexC', 'DewPointC', 'WindChillC', 'WindGustKmph', 'RES_LEVEL_FT']]
features.index = df['FLOW_DATE']
features.head()


# %%
features.plot(subplots = True)


# %%
# standardize the data
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis = 0)
data_std = dataset[:TRAIN_SPLIT].std(axis = 0)
# data_mean
data_std.shape
dataset = (dataset-data_mean)/data_std
dataset.shape


# %%
data_mean


# %%
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step) # step used here.
    data.append(dataset[indices])

    if single_step: # single_step used here.
      labels.append(target[i+target_size]) 
    else:
      labels.append(target[i:i+target_size]) 

  return np.array(data), np.array(labels)


# %%
dataset.shape


# %%
future_target = 90 # 90 future values
past_history = 600
print (dataset[:,13]) # water res levels needs to be predicted
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 13], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 13],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print(x_train_multi.shape)
print(y_train_multi.shape)
print(x_val_multi.shape)
print(y_val_multi.shape)


# %%
# TF DATASET

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


# %%
np.isnan(y_train_multi).any()


# %%
#utility function
def create_time_steps(length):
  return list(range(-length, 0))

print(create_time_steps(20))


# %%
#plotting function
def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)
  plt.grid()
  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()
  


for x, y in train_data_multi.take(1):
  multi_step_plot(x[0], y[0], np.array([0]))


# %%
print (x_train_multi.shape[-2:])
print (x_train_multi.shape)
# make the LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(16,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(8, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(90)) # for 90 outputs

# multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
# multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0), loss='mae')
multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')


multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=STEPS_PER_EPOCH,
                                          validation_data=val_data_multi,
                                          validation_steps=50)


# %%


# Plot train and validation loss over epochs

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.grid()

  plt.show()


# %%
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


# %%
print (multi_step_history)


# %%
for x, y in val_data_multi.take(5):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


# %%
# perform test evaluation


# %%



# %%



