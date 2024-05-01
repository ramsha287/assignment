# assignment
# assignment
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# a) Read test and label files
def read_files(label_file, test_file):
    test_data = pd.read_csv(test_file, header=None)  # Specify header=None to treat the first row as data
    label_data = pd.read_csv(label_file, header=None)  # Specify header=None to treat the first row as data
    return test_data, label_data

def draw_time_series_plots(test_data,label_data):
    # Read the multivariate time series data
    data = pd.read_csv("C:\\Users\\pc\\Desktop\\files\\test.csv")

    # Remove the timestamp column if present
    if 'timestamp' in data.columns:
        data.drop('timestamp', axis=1, inplace=True)

    # Calculate the rolling mean and standard deviation
    window_size = 10  # Adjust the window size as needed
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    # Calculate the upper and lower bounds for anomaly detection
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)

    # Find the anomalies
    anomalies = data[(data > upper_bound) | (data < lower_bound)]

    # Plot the time series with anomaly regions
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    for col in anomalies.columns:
        plt.scatter(anomalies.index, anomalies[col], color='red', label='Anomaly')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series Plot with Anomaly Regions')
    plt.legend()
    plt.show()


# c) Perform EDA and find out root cause
def perform_eda(test_data):
    eda_result = test_data.describe()  # Example: Basic statistics summary
    return eda_result


# d) Find out the variables which are the root cause for the anomaly
def find_root_cause(eda_result):
    # Example: Analyze statistics to identify root causes
    root_cause = eda_result['value']['mean'] + eda_result['value']['std']
    return root_cause


# Example usage:
if __name__ == "__main__":
    # Example file paths
    test_file = "C:\\Users\\pc\\Desktop\\files\\test.csv"
    label_file = "C:\\Users\\pc\\Desktop\\files\\test_label.csv"

    # a) Read test and label files
    test_data, label_data = read_files(test_file, label_file)
    # b) Draw time series plots with anomaly regions
    draw_time_series_plots(test_data, label_data)
    # c) Perform EDA and find out root cause
    eda_result = perform_eda(test_data)
    print("EDA Result:")
    print(eda_result)
    # d) Find out the variables which are the root cause for the anomaly
    root_cause = find_root_cause(eda_result)
    print("Root Cause for Anomaly:", root_cause)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# a) Read test and label files
def read_files(label_file, test_file):
    test_data = pd.read_csv(test_file, header=None)  # Specify header=None to treat the first row as data
    label_data = pd.read_csv(label_file, header=None)  # Specify header=None to treat the first row as data
    return test_data, label_data

def draw_time_series_plots(test_data,label_data):
    # Read the multivariate time series data
    data = pd.read_csv("C:\\Users\\pc\\Desktop\\files\\msl_test.csv")

    # Remove the timestamp column if present
    if 'timestamp' in data.columns:
        data.drop('timestamp', axis=1, inplace=True)

    # Calculate the rolling mean and standard deviation
    window_size = 10  # Adjust the window size as needed
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    # Calculate the upper and lower bounds for anomaly detection
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)

    # Find the anomalies
    anomalies = data[(data > upper_bound) | (data < lower_bound)]

    # Plot the time series with anomaly regions
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    for col in anomalies.columns:
        plt.scatter(anomalies.index, anomalies[col], color='red', label='Anomaly')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series Plot with Anomaly Regions')
    plt.legend()
    plt.show()


# c) Perform EDA and find out root cause
def perform_eda(test_data):
    eda_result = test_data.describe()  # Example: Basic statistics summary
    return eda_result


# d) Find out the variables which are the root cause for the anomaly
def find_root_cause(eda_result):
    # Example: Analyze statistics to identify root causes
    root_cause = eda_result['value']['mean'] + eda_result['value']['std']
    return root_cause


# Example usage:
if __name__ == "__main__":
    # Example file paths
    test_file = "C:\\Users\\pc\\Desktop\\files\\msl_test.csv"
    label_file = "C:\\Users\\pc\\Desktop\\files\\msl_test_label.csv"

    # a) Read test and label files
    test_data, label_data = read_files(test_file, label_file)
    # b) Draw time series plots with anomaly regions
    draw_time_series_plots(test_data, label_data)
    # c) Perform EDA and find out root cause
    eda_result = perform_eda(test_data)
    print("EDA Result:")
    print(eda_result)
    # d) Find out the variables which are the root cause for the anomaly
    root_cause = find_root_cause(eda_result)
    print("Root Cause for Anomaly:", root_cause)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# a) Read test and label files
def read_files(label_file, test_file):
    test_data = pd.read_csv(test_file, header=None)  # Specify header=None to treat the first row as data
    label_data = pd.read_csv(label_file, header=None)  # Specify header=None to treat the first row as data
    return test_data, label_data

def draw_time_series_plots(test_data,label_data):
    # Read the multivariate time series data
    data = pd.read_csv("C:\\Users\\pc\\Desktop\\files\\psm_test.csv")

    # Remove the timestamp column if present
    if 'timestamp' in data.columns:
        data.drop('timestamp', axis=1, inplace=True)

    # Calculate the rolling mean and standard deviation
    window_size = 10  # Adjust the window size as needed
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    # Calculate the upper and lower bounds for anomaly detection
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)

    # Find the anomalies
    anomalies = data[(data > upper_bound) | (data < lower_bound)]

    # Plot the time series with anomaly regions
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    for col in anomalies.columns:
        plt.scatter(anomalies.index, anomalies[col], color='red', label='Anomaly')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series Plot with Anomaly Regions')
    plt.legend()
    plt.show()


# c) Perform EDA and find out root cause
def perform_eda(test_data):
    eda_result = test_data.describe()  # Example: Basic statistics summary
    return eda_result


# d) Find out the variables which are the root cause for the anomaly
def find_root_cause(eda_result):
    # Example: Analyze statistics to identify root causes
    root_cause = eda_result['value']['mean'] + eda_result['value']['std']
    return root_cause


# Example usage:
if __name__ == "__main__":
    # Example file paths
    test_file = "C:\\Users\\pc\\Desktop\\files\\psm_test.csv"
    label_file = "C:\\Users\\pc\\Desktop\\files\\psm_test_label.csv"

    # a) Read test and label files
    test_data, label_data = read_files(test_file, label_file)
    # b) Draw time series plots with anomaly regions
    draw_time_series_plots(test_data, label_data)
    # c) Perform EDA and find out root cause
    eda_result = perform_eda(test_data)
    print("EDA Result:")
    print(eda_result)
    # d) Find out the variables which are the root cause for the anomaly
    root_cause = find_root_cause(eda_result)
    print("Root Cause for Anomaly:", root_cause)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# a) Read test and label files
def read_files(label_file, test_file):
    test_data = pd.read_csv(test_file, header=None)  # Specify header=None to treat the first row as data
    label_data = pd.read_csv(label_file, header=None)  # Specify header=None to treat the first row as data
    return test_data, label_data

def draw_time_series_plots(test_data,label_data):
    # Read the multivariate time series data
    data = pd.read_csv("C:\\Users\\pc\\Desktop\\files\\smap_test.csv")

    # Remove the timestamp column if present
    if 'timestamp' in data.columns:
        data.drop('timestamp', axis=1, inplace=True)

    # Calculate the rolling mean and standard deviation
    window_size = 10  # Adjust the window size as needed
    rolling_mean = data.rolling(window=window_size).mean()
    rolling_std = data.rolling(window=window_size).std()

    # Calculate the upper and lower bounds for anomaly detection
    upper_bound = rolling_mean + (2 * rolling_std)
    lower_bound = rolling_mean - (2 * rolling_std)

    # Find the anomalies
    anomalies = data[(data > upper_bound) | (data < lower_bound)]

    # Plot the time series with anomaly regions
    plt.figure(figsize=(10, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)

    for col in anomalies.columns:
        plt.scatter(anomalies.index, anomalies[col], color='red', label='Anomaly')

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series Plot with Anomaly Regions')
    plt.legend()
    plt.show()


# c) Perform EDA and find out root cause
def perform_eda(test_data):
    eda_result = test_data.describe()  # Example: Basic statistics summary
    return eda_result


# d) Find out the variables which are the root cause for the anomaly
def find_root_cause(eda_result):
    # Example: Analyze statistics to identify root causes
    root_cause = eda_result['value']['mean'] + eda_result['value']['std']
    return root_cause


# Example usage:
if __name__ == "__main__":
    # Example file paths
    test_file = "C:\\Users\\pc\\Desktop\\files\\smap_test.csv"
    label_file = "C:\\Users\\pc\\Desktop\\files\\smap_test_label.csv"

    # a) Read test and label files
    test_data, label_data = read_files(test_file, label_file)
    # b) Draw time series plots with anomaly regions
    draw_time_series_plots(test_data, label_data)
    # c) Perform EDA and find out root cause
    eda_result = perform_eda(test_data)
    print("EDA Result:")
    print(eda_result)
    # d) Find out the variables which are the root cause for the anomaly
    root_cause = find_root_cause(eda_result)
    print("Root Cause for Anomaly:", root_cause)
