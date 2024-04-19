import ast
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Declare variable
data_unprocessed_path = 'data/data_collected_unprocessed/'

# Config panda
pd.set_option('display.max_columns', None)


def convert_string_to_list(string):
    """
    Convert string into list
    :param string:
    :return: list of integer
    """
    int_list = [int(item) for item in string.strip("[']").split("', ' ")]
    return int_list


def removed_error_data(dataframe):
    """
    Removed row contains invalid value
    :param dataframe:
    :return: pandas DataFrame
    """
    a = "Error collecting data on middle finger"
    b = "Error collecting data on thumb finger"
    c = "Error collecting data on index finger"
    d = "'-1',' -1',' -1'"
    dataframe = dataframe[~(dataframe.Accel.str.contains(a)) | (dataframe.Gyr.str.contains(a))]
    dataframe = dataframe[~(dataframe.Accel.str.contains(b)) | (dataframe.Gyr.str.contains(b))]
    dataframe = dataframe[~(dataframe.Accel.str.contains(c)) | (dataframe.Gyr.str.contains(c))]
    dataframe = dataframe[~(dataframe.Accel.str.contains(d)) | (dataframe.Gyr.str.contains(d))]
    dataframe = dataframe[~(dataframe.Accel.str.contains("Average sensor values"))]
    return dataframe


def merge_csv_file(folder_path):
    """
    Merge all data collected into one big data frame
    :param folder_path: input folder
    :return: pandas DataFrame
    """
    # List to store individual DataFrame
    df_list = []


    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            dataframe.loc[0, "Sep"] = True
            # Data Encoding one hot encoding
            dataframe= pd.get_dummies(dataframe, columns=['Label'],dtype= float)

            # Append the df to the list
            df_list.append(dataframe)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


def convert_str_to_int(dataframe):
    """
    Convert the value in the Accel and Gyr from list of string to list of int
    :param dataframe: df need to change
    :return: pandas DataFrame
    """
    # Convert string values in lists to integers
    dataframe.Accel = dataframe.Accel.apply(lambda x: convert_string_to_list(x))
    dataframe.Gyr = dataframe.Gyr.apply(lambda x: convert_string_to_list(x))

    return dataframe


def fix_sep_value(dataframe):
    """
    Fix the duplicate True value, switching back to False.
    :param dataframe: df need to change
    :return: pandas DataFrame
    """
    # Find the indices where 'Sep' is True
    true_indices = dataframe.index[dataframe.Sep == True]

    # Iterate over the indices and update the value of 'Sep' to False for adjacent True values
    for idx in range(len(true_indices) - 1):
        if true_indices[idx + 1] - true_indices[idx] == 1:
            dataframe.at[true_indices[idx + 1], 'Sep'] = False
    return dataframe


def get_data():
    print("Merge dataset")
    df = merge_csv_file(data_unprocessed_path)
    print("Removed NaN and duplicated")
    df = df.dropna()
    df = df.drop_duplicates()
    print("Removed Error data")
    df = removed_error_data(df)
    print("Convert str to int")
    df = convert_str_to_int(df)
    print("Fix sep value")
    df = fix_sep_value(df)
    print("Done preprocessing data")
    return df


if __name__ == "__main__":
    pass
