import os
import re
import pandas as pd
import numpy as np

# Config panda
pd.set_option('display.max_columns', None)


def string_to_list(string):
    int_list = [float(item) for item in string]
    return int_list


# Function to check if any element in the list is a string
def contains_string(lst):
    converted_list = []
    for item in lst:
        if not re.search('[a-zA-Z]', item):
            try:
                # Attempt to convert each item to float
                converted_list.append(float(item))
            except ValueError:
                # If conversion fails, ignore the item
                continue
        else:
            continue
    if len(converted_list) == 12:
        return False


def convert_json_to_dataframe():
    # List to store individual DataFrame
    df_list = []

    for filename in os.listdir('data/new_data'):
        if filename.endswith('.json'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join('data/new_data', filename)
            dataframe = pd.read_json(file_path)

            # Append the df to the list
            df_list.append(dataframe)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


def filter_position_value(dataframe):
    dataframe = dataframe[dataframe.position.apply(len) == 12]
    dataframe = dataframe[dataframe.position.apply(contains_string) == False]
    dataframe.position = dataframe.position.apply(lambda x: string_to_list(x))
    return dataframe


def get_data():
    df = convert_json_to_dataframe()
    df = filter_position_value(df)
    X = df.drop(['timestamp', 'label'], axis=1)
    total = []
    for index, row in X.iterrows():
        total.append(row['position'])
    X = np.array(total)
    y = df['label']
    y = pd.get_dummies(y, columns=['label'], dtype=float)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    get_data()
