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
    int_list = [int(item) for item in string.strip("[]").strip("'").split("', ' ")]
    return int_list


def removed_error_data(string):
    a = "Error collecting data on middle finger"
    b = "Error collecting data on thumb finger"
    c = "Error collecting data on index finger"
    if a in string:
        return string.strip(a)
    return string


def merge_csv_file(folder_path):
    """
    Merge all data collected into one big data frame
    :param folder_path: input folder
    :return: panda DataFrame
    """
    # List to store individual DataFrame
    df_list = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            # Append the df to the list
            df_list.append(dataframe)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


def convert_str_to_int(dataframe):
    """
    Convert the value in the Accel and Gyr from list of string to list of int
    :param dataframe: df need to change
    :return: panda DataFrame
    """
    # Convert string values in lists to integers
    dataframe.Accel = dataframe.Accel.apply(lambda x: convert_string_to_list(x))
    dataframe.Gyr = dataframe.Gyr.apply(lambda x: convert_string_to_list(x))

    return dataframe


def fix_sep_value(dataframe):
    """
    Fix the duplicate True value, switching back to False.
    :param dataframe: df need to change
    :return: pd DataFrame
    """
    # Find the indices where 'Sep' is True
    true_indices = dataframe.index[dataframe.Sep == True]

    # Iterate over the indices and update the value of 'Sep' to False for adjacent True values
    for idx in range(len(true_indices) - 1):
        if true_indices[idx + 1] - true_indices[idx] == 1:
            dataframe.at[true_indices[idx + 1], 'Sep'] = False
    return dataframe


def fix_add_sep_for_pinch2finger(dataframe):
    """
    Fix the dataframe by adding a True sep for each pinch2finger
    :param dataframe: df need to change
    :return: pd DataFrame
    """
    # Find the indices where the label is 'pinch2finger'
    pinch2finger_indices = dataframe.index[dataframe['Label'] == 'pinch2finger']

    # Iterate over the indices and update the value of 'Sep' to True only for the first occurrence
    for idx in pinch2finger_indices:
        # Check if the index is greater than 0 and the previous row's label is not 'pinch2finger'
        if idx > 0 and dataframe.at[idx - 1, 'Label'] != 'pinch2finger':
            dataframe.at[idx, 'Sep'] = True

    return dataframe


def get_data():
    print("Merge dataset")
    df = merge_csv_file(data_unprocessed_path)
    print("Removed NaN and duplicated")
    df = df.dropna()
    df = df.drop_duplicates()
    print("Convert str to int")
    df = convert_str_to_int(df)
    print("Fix sep value")
    df = fix_sep_value(df)
    df = fix_add_sep_for_pinch2finger(df)
    print("Done preprocessing data")

    return df


if __name__ == "__main__":
    pass
