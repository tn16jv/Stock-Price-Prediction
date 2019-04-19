import numpy as np


def normalize(df, target_cols, min, max):
    """
    Normalizes a Pandas dataframe values to the range of [-1, 1]
    :param min: the inclusive minimum
    :param max: the inclusive maximum
    :return:
    :param df: Pandas dataframe or object
    :param target_cols: Array of strings that holds the headers to normalize
    :return: Normalized dataframe
    """
    #survey_data[cols_to_norm] = survey_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    result = df.copy()
    for column in df.columns:
        if column in target_cols:
            max_value = df[column].max()
            min_value = df[column].min()
            result[column] = ((max-min) * (df[column] - min_value)
                              / (max_value - min_value) + min)
    return result


def average_df(df, target_cols):
    result = df[target_cols.copy()]
    return result.mean(axis=1)


def df_to_array(df, target_cols):
    """
    Creates an numpy array representation of a Pandas dataframe
    :param df:
    :param target_cols:
    :return:
    """
    result = []
    for column in df.columns:
        if column in target_cols:
            result.append(df[column])
    return np.array(result)


def get_X(df):
    step_size = 1
    X = []
    for i in range(len(df)-step_size-1):
        data = df[i:(i+step_size), 0]
        X.append(data)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X

def get_Y(df):
    step_size = 1
    Y = []
    for i in range(len(df) - step_size - 1):
        data = df[i:(i + step_size), 0]
        Y.append(df[i + step_size, 0])
    Y = np.array(Y)
    return Y


def train_test_Y(data, percentage):
    '''Not used, for future reference'''
    data = data.values
    split = int(len(data) * percentage)
    training = data[:split]
    testing = data[split:]
    return training, testing


def create_training_sets(df, seq_len, valid_percentage=0.10, test_percentage=0.10):
    data = []
    df_data = df.values
    for i in range(len(df_data) - seq_len):
        data.append(df_data[i: i + seq_len])    # add in chunks of seq_len
    data = np.array(data)

    valid_size = int(np.round(valid_percentage * data.shape[0]))
    test_size = int(np.round(test_percentage * data.shape[0]))
    #train_size = data.shape[0] - (test_size + valid_size)
    train_size = data.shape[0] - test_size

    # triple list indices for each dimension of the 3d arrays
    x_train = data[:train_size, :-1,:]  # 1st dimension is train_size length, 2nd is total-1, and 3rd is all
    y_train = data[:train_size, -1, :]  # 1st dimension gets train_size length, no 2nd dimension here, and 3rd is all

    #x_valid = data[train_size: train_size + valid_size, :-1, :]
    #y_valid = data[train_size: train_size+valid_size, -1, :]
    x_valid = []
    y_valid = []

    #x_test = data[train_size + valid_size:, :-1, :]
    #y_test = data[train_size + valid_size:, -1, :]
    x_test = data[train_size:, :-1, :]
    y_test = data[train_size:, -1, :]

    return x_train, y_train, x_valid, y_valid, x_test, y_test
