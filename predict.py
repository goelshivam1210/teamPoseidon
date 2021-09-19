import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import SaveableModel, dataset_to_arrays

FORECAST_LENGTH = 90

HISTORY_LENGTH = 10
STEP = 1 # for creation of dataset

used_cols = ['PRESENT_STORAGE_TMC', 'INFLOW_CUSECS', 'OUTFLOW_CUECS', 'tempC', 'windspeedKmph', 'precipMM', 'humidity', 'pressure (mB)', 'WindChillC', 'RES_LEVEL_FT', 'OUTFLOW_CUECS_HEMAVATHI', 'OUTFLOW_CUECS_HARANGI']


data_dir = Path("data")
data_file = data_dir / "updated_data_set_full.csv"
data_dict_file = data_dir / "data_dict.pkl"
future_data_file = data_dir / "future_dataset.pkl"

model_dir = Path("saved_models")
model_file = model_dir / "model.pkl"
ub_model_file = model_dir / "ub_model.pkl"
lb_model_file = model_dir / "lb_model.pkl"

plot_dir = Path("plots")

prediction_dir = Path("predictions")
prediction_file = prediction_dir / "future_predictions_updated_model_weights_sep-nov.csv"

# load formatted data dictionary
with open(data_dict_file, "rb") as f:
    data_dict = pickle.load(f)

# load models
with open(model_file, "rb") as f:
    mid_model = pickle.load(f)

with open(ub_model_file, "rb") as f:
    ub_model = pickle.load(f)

with open(lb_model_file, "rb") as f:
    lb_model = pickle.load(f)

def create_time_steps(length):
    return list(range(-length, 0))

#plotting function
def multi_step_plot(history, true_future, prediction, title=""):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.grid()
    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                label='Predicted Future')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

# evaluation
def generate_plots_for_partition(partition_name):
    print(f"generating plots for {partition_name}")
    plt.figure(figsize=(20,10))
    plt.style.use('tableau-colorblind10')
    xs = data_dict[partition_name]["X"]
    ys = data_dict[partition_name]["y"]
    ypred_lb = lb_model.predict(xs)
    ypred_mid = mid_model.predict(xs)
    ypred_ub = ub_model.predict(xs)

    num_past = xs.shape[0]

    num_future = ypred_lb.shape[0]

    x_indices_future = data_dict[partition_name]["dates"] #np.arange(num_future)
    plt.figure(figsize=(20,10))
    ticks = plt.xticks(fontsize = 20)
    ticks = plt.yticks(fontsize = 22)
    # plot future predictions
    plt.fill_between(x_indices_future, ypred_lb, ypred_ub, color='b', alpha=0.1, label = 'Predicted Lower/Upper Bound')
    plt.plot(x_indices_future, ypred_mid, "--", label = 'Predicted Value')
    plt.plot(x_indices_future, ys, label = 'True Value')
    plt.legend(fontsize = 22)
    plt.title(f"{partition_name} Evaluation", fontdict = {'fontsize' : 36})

    plt.savefig(plot_dir / f"{partition_name}_results.png", dpi = 800)
    plt.close()

generate_plots_for_partition("train")
generate_plots_for_partition("validation")
generate_plots_for_partition("test")

with open(future_data_file, "rb") as f:
    future_xs = pickle.load(f)

partition_name = "future"

# d = data_dict[partition_name]
xs = future_xs[-90:]

ypred_lb = lb_model.predict(xs)
ypred_mid = mid_model.predict(xs)
ypred_ub = ub_model.predict(xs)

num_past = xs.shape[0]

num_future = ypred_mid.shape[0]

x_indices_future = np.arange(num_future)
plt.figure(figsize=(20,10))

# plot future predictions
plt.plot(x_indices_future, ypred_mid, "--", label = "Predicted Value")

plt.title("Future Forecasts", fontdict = {'fontsize' : 26})
plt.style.use('tableau-colorblind10')
plt.legend(prop={"size":20})
plt.grid(True)
plt.savefig(plot_dir / "future_forecast.png", dpi = 1200)
plt.close()

data_to_dump = {
    "mid":ypred_mid,
    "lb": ypred_lb,
    "ub":ypred_ub,
}

pd.DataFrame(data_to_dump).to_csv(prediction_file, index=False)


pd.to_timedelta(FORECAST_LENGTH, unit="day")

feature_names = [f"{c} d-{i}" for i in range(HISTORY_LENGTH-1, -1, -1) for c in used_cols]
feature_importance_df = pd.DataFrame({"feature":feature_names, "importance":mid_model.feature_importances_})

print(feature_importance_df.sort_values("importance", ascending=False).tail(20))

