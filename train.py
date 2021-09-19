import os
from pathlib import Path

from sklearn import ensemble, metrics
import matplotlib.pyplot as plt
from fast_ml import model_development as md
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

from model_utils import SaveableModel

# pathing
data_dir = Path("data")
data_file = data_dir / "updated_data_set_full.csv"
data_dict_file = data_dir / "data_dict.pkl"
future_data_file = data_dir / "future_dataset.pkl"

model_dir = Path("saved_models")
model_file = model_dir / "model.pkl"
ub_model_file = model_dir / "ub_model.pkl"
lb_model_file = model_dir / "lb_model.pkl"

## Hypers
SPLITS = {
    "train":0.70,
    "validation":0.10,
    "test":0.20,
}

# they definitely needs optimization.
BATCH_SIZE = 64 # bacth size in batch-SGD/variants
BUFFER_SIZE = 10 # for shuffling the dataset
STEP = 1 # for creation of dataset

# Train and evaluate
STEPS_PER_EPOCH = 10 # hyperparameter
EPOCHS = 100 # hyperparameter

FORECAST_LENGTH = 90

HISTORY_LENGTH = 10


# data loading
df = pd.read_csv(data_file, sep=',', na_values=[" ", "&nbsp;"], parse_dates=["FLOW_DATE"], header=0)
df = df.ffill()
df = df.set_index("FLOW_DATE")
df["dummy"] = 1

used_cols = ['PRESENT_STORAGE_TMC', 'INFLOW_CUSECS', 'OUTFLOW_CUECS', 'tempC', 'windspeedKmph', 'precipMM', 'humidity', 'pressure (mB)', 'WindChillC', 'RES_LEVEL_FT', 'OUTFLOW_CUECS_HEMAVATHI', 'OUTFLOW_CUECS_HARANGI']

target_col = 'RES_LEVEL_FT'

def get_windowed_dataset(
    dataframe,
    target_col,
    history_length=HISTORY_LENGTH,
    forecast_length=FORECAST_LENGTH,
    batch_size=32,
    stride=30,
    ):

    data = dataframe.values.copy()
    ys = dataframe[target_col]

    data_future = data.copy()
    data = data[:-forecast_length-history_length]
    ys = ys[forecast_length+history_length:] # predict window starting at next time step

    # window the labels to make forecasts in the future
    windowed_ys = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=ys,
        targets=None, # ys.index,
        sequence_length=forecast_length,
        sequence_stride=stride,
        shuffle=False,
        batch_size=1,
    )
    ys = np.array(list(windowed_ys.as_numpy_iterator())).squeeze()
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=ys,
        sequence_length=history_length,
        sequence_stride=stride,
        shuffle=False,
        batch_size=batch_size,
    )
    future_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data_future,
        targets=np.ones((data_future.shape[0],1)),
        sequence_length=history_length,
        sequence_stride=stride,
        shuffle=False,
        batch_size=batch_size,
    )
    return dataset, future_dataset

def dataset_to_arrays(dataset, flatten=True, last_y=True):
    all_xs = []
    all_ys = []
    for x, y in dataset.as_numpy_iterator():
        all_xs.append(x)
        all_ys.append(y)
    ys = np.concatenate(all_ys)
    xs = np.concatenate(all_xs)
    if flatten:
        xs = xs.reshape(xs.shape[0], -1)
    if last_y:
        ys = ys[:, -1]
    return xs, ys

historical_dataset, future_dataset  = get_windowed_dataset(
    df[used_cols],# + ["dummy"]],
    target_col=target_col,
    stride=1
    )

historical_xs, historical_ys = dataset_to_arrays(historical_dataset)

# series of y's which are indexed by date
historical_ys_series = df[target_col].iloc[-historical_ys.shape[0]:]
assert (historical_ys_series == historical_ys).all(), "something went wrong with historical y indexes"

future_xs, _ = dataset_to_arrays(future_dataset)

with open(future_data_file, "wb") as f:
    pickle.dump(future_xs, f)

data_df = pd.DataFrame(np.concatenate([np.arange(historical_ys.shape[0])[:, None], historical_xs, historical_ys[:,None]], axis=-1))

data_df.index = historical_ys_series.index

X_train, y_train, X_valid, y_valid, X_test, y_test = md.train_valid_test_split(
    data_df,
    target=data_df.columns[-1], # use dummy for train test split before preproccessing
    train_size=SPLITS["train"],
    valid_size=SPLITS["validation"],
    test_size=SPLITS["test"],
    method="sorted",
    sort_by_col=data_df.columns[0],
)

# use .iloc[:,1:]to drop first column
# the dataframes are indexed by the date of the y value (the future flow)
data_dict = {
    "train": {"X":X_train.iloc[:,1:], "y":y_train, "dates": y_train.index},
    "validation": {"X":X_valid.iloc[:,1:], "y":y_valid, "dates": y_valid.index},
    "test": {"X":X_test.iloc[:,1:], "y":y_test, "dates": y_test.index},
}

with open(data_dict_file, "wb") as f:
    pickle.dump(data_dict, f)

X_means = data_dict["train"]["X"].mean()
X_stdev = data_dict["train"]["X"].std()
y_means = data_dict["train"]["y"].mean()
y_stdev = data_dict["train"]["y"].std()
for partition in data_dict.keys():
    data_dict[partition]["X"] = (data_dict[partition]["X"] - X_means) / X_stdev
    data_dict[partition]["y"] = (data_dict[partition]["y"] - y_means) / y_stdev

future_xs = (future_xs - X_means[None,]) / X_stdev[None,:]

# climate that gets the most rainfall, months of may to october get most rainfall
monsoon_weight = 20
for part, d in data_dict.items():
    data_dict[part]["is_monsoon_month"] = (5 <= d["dates"].month) & (d["dates"].month <= 10)
    data_dict[part]["sample_weight"] = np.ones_like(data_dict[part]["is_monsoon_month"]).astype(float)
    data_dict[part]["sample_weight"][data_dict[part]["is_monsoon_month"]] = monsoon_weight

kwargs = dict(
    max_depth=20,
    subsample=0.5, n_estimators = 200
)

lower_model = ensemble.GradientBoostingRegressor(loss="quantile",alpha=0.1,**kwargs,)
mid_model = ensemble.GradientBoostingRegressor(loss="ls", **kwargs)
upper_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.9, **kwargs)

train_args = {
    "X":data_dict["train"]['X'],
    "y":data_dict["train"]["y"],
    "sample_weight":data_dict["train"]["sample_weight"]
}
lower_model.fit(**train_args)
mid_model.fit(**train_args)
upper_model.fit(**train_args)

model = SaveableModel(y_means, y_stdev, mid_model)
lb_model = SaveableModel(y_means, y_stdev, lower_model)
ub_model = SaveableModel(y_means, y_stdev, upper_model)


result_dict = {}
for partition, d in data_dict.items():
    rescale_y_pred = model.predict(d["X"])
    rescale_y_true = model.rescale(d["y"])
    mae = metrics.mean_absolute_error(rescale_y_true, rescale_y_pred)
    monsoon_mae = metrics.mean_absolute_error(rescale_y_true[d["is_monsoon_month"]], rescale_y_pred[d["is_monsoon_month"]])
    non_monsoon_mae = metrics.mean_absolute_error(rescale_y_true[~d["is_monsoon_month"]], rescale_y_pred[~d["is_monsoon_month"]])
    result_dict[partition] = {
        "mae":mae,
        "monsoon_mae": monsoon_mae,
        "non_monsoon_mae": non_monsoon_mae
    }
    print(f"{partition} mae:", mae)
    print(f"{partition} monsoon mae:", monsoon_mae)
    print(f"{partition} non monsoon mae:", non_monsoon_mae)
    # multi_step_plot(xs[:, -1], ys, y_pred, title=f"{partition}")

print("reslts")
print(pd.DataFrame(result_dict))

# save predictive model, i.e. y_stdev, y_means, lower_model, mid_model, upper_model

historical_xs.shape, X_means.shape

# refit model on full historical dataset and predict on future dataset

historical_training_xs = (historical_xs - X_means[None,]) / X_stdev[None,]
historical_training_ys = (historical_ys - y_means) / y_stdev

lower_model = ensemble.GradientBoostingRegressor(loss="quantile",alpha=0.1,**kwargs,)
mid_model = ensemble.GradientBoostingRegressor(loss="ls", **kwargs)
upper_model = ensemble.GradientBoostingRegressor(loss="quantile", alpha=0.9, **kwargs)

lower_model.fit(historical_training_xs, historical_training_ys)
mid_model.fit(historical_training_xs, historical_training_ys)
upper_model.fit(historical_training_xs, historical_training_ys)

model = SaveableModel(y_means, y_stdev, mid_model)
lb_model = SaveableModel(y_means, y_stdev, lower_model)
ub_model = SaveableModel(y_means, y_stdev, upper_model)

print("training on full historical dataset and saving models to file")
with open(model_file, "wb") as f:
    pickle.dump(model, f)

with open(lb_model_file, "wb") as f:
    pickle.dump(lb_model, f)

with open(ub_model_file, "wb") as f:
    pickle.dump(ub_model, f)