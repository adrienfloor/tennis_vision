# Generate a new X

import pandas as pd
from sktime.datatypes._panel._convert import from_2d_array_to_nested

# Generate X with 10 lags per sequence
# -----------------------------------------------------------------------------#

def generate_X10(df):
    #Generation of lags for df shape(x,y,V). Take care ! Index needed !
    data = pd.DataFrame()
    for i in range(10, 0, -1):
        data[f'lagX_{i}'] = df['x'].shift(i, fill_value=0)
        data[f'lagY_{i}'] = df['y'].shift(i, fill_value=0)
        data[f'lagV_{i}'] = df['V'].shift(i, fill_value=0)

    #Generation of sequences
    Xs = data[[
        'lagX_10','lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1'
    ]]

    Xs = from_2d_array_to_nested(Xs.to_numpy()) # <- pack all lags in one sequence per row

    Ys = data[[
            'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
            'lagY_3', 'lagY_2', 'lagY_1'
    ]]

    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = data[[
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
            'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
    ]]

    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs],1) # <- !!! X.columns => ['0','0','0'] <- Concatenator() error if same index

    X.columns=["X", "Y", "V"]

    return X

# -----------------------------------------------------------------------------#
#Generate X with 20 lags per sequence

def generate_X20(df):
    #Generation of lags for df shape(x,y,V). Take care ! Index needed !
    data = pd.DataFrame()
    for i in range(20, 0, -1):
        data[f'lagX_{i}'] = df['x'].shift(i, fill_value=0)
        data[f'lagY_{i}'] = df['y'].shift(i, fill_value=0)
        data[f'lagV_{i}'] = df['V'].shift(i, fill_value=0)

    #Generation of sequences
    Xs = data[[
        'lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11',
        'lagX_10', 'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6',
        'lagX_5', 'lagX_4', 'lagX_3','lagX_2', 'lagX_1'
    ]]

    Xs = from_2d_array_to_nested(Xs.to_numpy()) # <- pack all lags in one sequence per row

    Ys = data[[
        'lagY_20', 'lagY_19', 'lagY_18', 'lagY_17','lagY_16',
        'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6',
        'lagY_5', 'lagY_4', 'lagY_3', 'lagY_2', 'lagY_1'
    ]]

    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = data[[
        'lagV_20', 'lagV_19', 'lagV_18','lagV_17', 'lagV_16',
        'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12','lagV_11',
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6',
        'lagV_5', 'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
    ]]

    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs], 1) # <- !!! X.columns => ['0','0','0'] <- Concatenator() error if same index

    X.columns=["X", "Y", "V"]

    return X

# -----------------------------------------------------------------------------#

#Generate X,y train 10lags per sequence

def generate_data10_train(df):
    #Generation of lags for df shape(x,y,V). Take care ! Index needed !

    id_bounces = df.index[df.bounce==1]
    for i in id_bounces:
        df.bounce.iloc[i+1:i+10]=1
        y_train = df.bounce

    df = df.drop('bounce', axis=1)

    data = pd.DataFrame()
    for i in range(10, 0, -1):
        data[f'lagX_{i}'] = df['x'].shift(i, fill_value=0)
        data[f'lagY_{i}'] = df['y'].shift(i, fill_value=0)
        data[f'lagV_{i}'] = df['V'].shift(i, fill_value=0)

    #Generation of sequences
    Xs = data[[
        'lagX_10','lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1'
    ]]

    Xs = from_2d_array_to_nested(Xs.to_numpy()) # <- pack all lags in one sequence per row

    Ys = data[[
            'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
            'lagY_3', 'lagY_2', 'lagY_1'
    ]]

    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = data[[
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
            'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
    ]]

    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs],1) # <- !!! X.columns => ['0','0','0'] <- Concatenator() error if same index

    X.columns=["X", "Y", "V"]

    return X, y_train

# -----------------------------------------------------------------------------#

#Generate X,y train 10lags per sequence

def generate_data20_train(df):
    #Generation of lags for df shape(x,y,V). Take care ! Index needed !

    id_bounces = df.index[df.bounce==1]
    for i in id_bounces:
        df.bounce.iloc[i+1:i+20]=1
        y_train = df.bounce

    df = df.drop('bounce', axis=1)

    data = pd.DataFrame()
    for i in range(20, 0, -1):
        data[f'lagX_{i}'] = df['x'].shift(i, fill_value=0)
        data[f'lagY_{i}'] = df['y'].shift(i, fill_value=0)
        data[f'lagV_{i}'] = df['V'].shift(i, fill_value=0)

    #Generation of sequences
    Xs = data[[
        'lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11',
        'lagX_10', 'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6',
        'lagX_5', 'lagX_4', 'lagX_3','lagX_2', 'lagX_1'
    ]]

    Xs = from_2d_array_to_nested(Xs.to_numpy()) # <- pack all lags in one sequence per row

    Ys = data[[
        'lagY_20', 'lagY_19', 'lagY_18', 'lagY_17','lagY_16',
        'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6',
        'lagY_5', 'lagY_4', 'lagY_3', 'lagY_2', 'lagY_1'
    ]]

    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vs = data[[
        'lagV_20', 'lagV_19', 'lagV_18','lagV_17', 'lagV_16',
        'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12','lagV_11',
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6',
        'lagV_5', 'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
    ]]

    Vs = from_2d_array_to_nested(Vs.to_numpy())

    X = pd.concat([Xs, Ys, Vs], 1) # <- !!! X.columns => ['0','0','0'] <- Concatenator() error if same index

    X.columns=["X", "Y", "V"]

    return X, y_train

#Generate train/test/split for dataframe lags

def tts_lags(df):
    nb_test = 800
    X_train = df.iloc[:-nb_test, :-1]
    y_train = df.iloc[:-nb_test, -1]
    X_test = df.iloc[-nb_test:, :-1]
    y_test = df.iloc[-nb_test:, -1]


#Generate train/test/split for timeseries df

def tts_timeseries(X,y):
    nb_test = 800
    X_train = X.iloc[:-nb_test, :]
    y_train = y.iloc[:-nb_test]
    X_test = X.iloc[:nb_test-10, :]
    y_test = y.iloc[:nb_test-10]

    return X_train, X_test, y_train, y_test

#Generate timeseries10lags + train/test/split

def get_timeseries_and_tts(df):
    X, y = generate_data10_train(df)

    X_train, X_test, y_train, y_test = tts_timeseries(X,y)

    return X_train, X_test, y_train, y_test
