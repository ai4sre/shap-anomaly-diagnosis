import os
import json
import time
import numpy as np
import pandas as pd
import shappack
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

## Parameters ###################################################
FILE_PATH = "./data/2021-08-18-argowf-chaos-b2qdj-user_pod-memory-hog_0.json"
PLOTS_NUM = 120
TARGET_METRICS = ["cpu_usage_seconds_total",
                  "memory_working_set_bytes",
                  "network_transmit_bytes_total",
                  "network_receive_bytes_total",
                  "fs_writes_total",
                  "fs_reads_total"]
PARAMS = {
    "n_components": 0.8
}
ANALYSIS_PERIOD = 20
N_WORKERS = 1
SEED = 123
np.random.seed(SEED)
#################################################################

class ShapPCA(object):
    def __init__(self, train_data, model=PCA(n_components=0.80)):
        self.model = model.fit(train_data)

    def predict(self, data):
        input_data = np.asarray(data)
        output_data = self._reconstruct_data(input_data)
        errors = np.mean((input_data - output_data) ** 2, axis=1)
        return np.asarray(errors)

    def reconstruction_error(self, data):
        input_data = np.asarray(data)
        output_data = self._reconstruct_data(input_data)
        recon_error = (input_data - output_data) ** 2
        return recon_error

    def _reconstruct_data(self, data):
        transformed_data = self.model.transform(data)
        reconstructed_data = self.model.inverse_transform(transformed_data)
        return reconstructed_data


def read_file(file_path):
    with open(file_path) as f:
        raw_data = json.load(f)
    containers_data = raw_data["containers"]
    data_df = pd.DataFrame()
    for con in containers_data:
        if con in ["queue-master", "rabbitmq", "session-db"]:
            continue
        for metric in containers_data[con]:
            container_name = metric["container_name"]
            metric_name = metric["metric_name"].replace("container_", "")
            if metric_name not in TARGET_METRICS:
                continue
            column_name = "{}_{}".format(container_name, metric_name)
            data_df[column_name] = np.array(metric["values"], dtype=np.float)[:, 1][-PLOTS_NUM:]
    data_df = data_df.round(4).fillna(data_df.mean())
    return data_df


def preprocessing(data_df):
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data_df)
    return data_std

if __name__ == '__main__':
    data_df = read_file(FILE_PATH)
    data_df = preprocessing(data_df)
    train_data, test_data = data_df[:-ANALYSIS_PERIOD], data_df[-ANALYSIS_PERIOD:]
    start = time.time()
    model = ShapPCA(train_data, model=PCA(n_components=PARAMS["n_components"]))
    time_train = round(time.time() - start, 6)
    print(f"Training: {time_train}")
    start = time.time()
    explainer = shappack.KernelExplainer(model.predict, train_data)
    shap_value = explainer.shap_values(test_data, n_workers=N_WORKERS)
    time_shap = round(time.time() - start, 3)
    print(f"SHAP: {time_shap}")
