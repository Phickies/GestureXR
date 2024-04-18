import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# Declare variable
data_unprocessed_path = './data_collected_unprocessed/'
data_all_frame_path = "./data_all_frame_csv/"


# Config panda
pd.set_option('display.max_columns', None)


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
            df = pd.read_csv(file_path)

            # Append the df to the list
            df_list.append(df)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


def convert_str_to_int(dataframe):
    """
    Convert the value in the Accel and Gyr from list of string to list of int
    :param dataframe: df need to change
    :return: panda DataFrame
    """
    # Removed invalid data, duplicated data
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(inplace=True)

    # Convert string values in lists to integers
    dataframe.Accel = dataframe.Accel.apply(eval)
    dataframe.Gyr = dataframe.Gyr.apply(eval)

    return dataframe


if __name__ == "__main__":

    # Merge all dataset into one big data
    output_path = os.path.join(data_all_frame_path, "data.csv")
    if not os.path.exists(data_all_frame_path):
        os.makedirs(data_all_frame_path)
        big_data = merge_csv_file(data_unprocessed_path)
        big_data = convert_str_to_int(big_data)
        big_data.to_csv(output_path, index=False)

    # Get data
    df = pd.read_csv(output_path)


