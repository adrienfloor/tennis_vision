# Generate a new X

import pandas as pd
from sktime.datatypes._panel._convert import from_2d_array_to_nested, from_nested_to_2d_array

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

def filter_fake_bounces(df):
    print('')
    print('')
    print('filter_fake_bounces')
    print('')
    print('')
    print(df.columns)
    print('')
    print('')
    print(type(df))
    print('')
    print('')
    print(df)
    for index, row in df.iterrows():
        #selecting from iteration starting at index 3, stopping before last 3 indexes, and when bounce
        if index >=3 and index <= (df.shape[0] - 3) and row['bounce'] == 1.0:
            #selecting last 3 rows
            print('')
            print('')
            print(df.iloc[index-1]['y'])
            print(type(df.iloc[index-1]['y']))
            print('')
            print('')
            last_3_rows = df.iloc[index-1]['y'] + df.iloc[index-2]['y'] + df.iloc[index-3]['y']
            #selecting next 3 rows
            next_3_rows = df.iloc[index+1]['y'] + df.iloc[index+2]['y'] + df.iloc[index+3]['y']
            # diff of direction between two predicted bounces
            diff = last_3_rows - next_3_rows
            # if y at index -1 is greater than y at index - 3 it means we're going from top to bottom of the court
            if df.iloc[index-1]['y'] > df.iloc[index-3]['y']:
                # in that direction, a diff greater than or equal to zero means a change of direction
                # we can conclude that it is a volley and not a bounce
                if diff >= 0:
                    # so we change it to "not bounce"
                    # data.iloc[index]['bounce'] = 0
                    df.loc[index, 'bounce'] = 0
            # if y at index -1 is greater than y at index - 3 it means we're going from bottom to top of the court
            elif df.iloc[index-1]['y'] < df.iloc[index-3]['y']:
                # in that direction, a diff less than or equal to zero means a change of direction
                # we can conclude that it is a volley and not a bounce
                if diff <= 0:
                    # data.iloc[index]['bounce'] = 0
                    df.loc[index, 'bounce'] = 0
        else:
            pass
    return df

def reverse_data(df):
    #Reverse sequences
    Xs = from_nested_to_2d_array(df.X)
    Ys = from_nested_to_2d_array(df.Y)
    Vs = from_nested_to_2d_array(df.V)
    #Redefine lags
    Xs.columns = [
        'lagX_10','lagX_9', 'lagX_8', 'lagX_7', 'lagX_6',
        'lagX_5', 'lagX_4', 'lagX_3', 'lagX_2', 'lagX_1'
    ]
    Ys.columns = [
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6',
        'lagY_5', 'lagY_4','lagY_3', 'lagY_2', 'lagY_1'
        ]
    Vs.columns = [
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6',
        'lagV_5', 'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
        ]

    #Concat
    X_bis = pd.concat([Xs,Ys,Vs], axis=1)

    #Reorder lags
    X_bis = X_bis[['lagX_10','lagY_10','lagV_10', 'lagX_9',
        'lagY_9','lagV_9', 'lagX_8','lagY_8','lagV_8', 'lagX_7',
        'lagY_7','lagV_7', 'lagX_6','lagY_6','lagV_6', 'lagX_5',
        'lagY_5','lagV_5', 'lagX_4','lagY_4','lagV_4', 'lagX_3',
        'lagY_3','lagV_3','lagX_2','lagY_2','lagV_2', 'lagX_1',
        'lagY_1','lagV_1'
    ]]

    X_final = pd.DataFrame()
    for i in range(10, 0, -1):
        X_final['x'] =  X_bis[f'lagX_{i}'].shift(i,fill_value=0)
        X_final['y'] =  X_bis[f'lagY_{i}'].shift(i, fill_value=0)
        X_final['v'] =  X_bis[f'lagV_{i}'].shift(i,fill_value=0)

    X_final.drop(index=[0,1],axis=0)

    return X_final

def reverse_data_20(df):
    #Reverse sequences
    Xs = (df.X)
    Ys = from_nested_to_2d_array(df.Y)
    Vs = from_nested_to_2d_array(df.V)
    #Redefine lags
    Xs.columns = [
        'lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11',
        'lagX_10','lagX_9', 'lagX_8', 'lagX_7', 'lagX_6',
        'lagX_5', 'lagX_4', 'lagX_3', 'lagX_2', 'lagX_1'
    ]
    Ys.columns = [
        'lagY_20', 'lagY_19', 'lagY_18', 'lagY_17','lagY_16',
        'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6',
        'lagY_5', 'lagY_4','lagY_3', 'lagY_2', 'lagY_1'
        ]
    Vs.columns = [
        'lagV_20', 'lagV_19', 'lagV_18','lagV_17', 'lagV_16',
        'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12','lagV_11',
        'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6',
        'lagV_5', 'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1'
        ]

    #Concat
    X_bis = pd.concat([Xs,Ys,Vs], axis=1)

    #Reorder lags
    X_bis = X_bis[['lagX_20','lagY_20','lagV_20', 'lagX_19','lagY_19','lagV_19', 'lagX_18','lagY_18','lagV_18', 'lagX_17','lagY_17','lagV_17', 'lagX_16','lagY_16','lagV_16',
       'lagX_15','lagY_15','lagV_15', 'lagX_14','lagY_14','lagV_14', 'lagX_13','lagY_13','lagV_13', 'lagX_12','lagY_12','lagV_12', 'lagX_11','lagY_11','lagV_11', 'lagX_10','lagY_10','lagV_10',
       'lagX_9','lagY_9','lagV_9', 'lagX_8','lagY_8','lagV_8', 'lagX_7','lagY_7','lagV_7', 'lagX_6','lagY_6','lagV_6', 'lagX_5','lagY_5','lagV_5', 'lagX_4','lagY_4','lagV_4', 'lagX_3','lagY_3','lagV_3',
       'lagX_2','lagY_2','lagV_2', 'lagX_1','lagY_1','lagV_1']
    ]

    X_final = pd.DataFrame()
    for i in range(20, 0, -1):
        X_final['x'] =  X_bis[f'lagX_{i}'].shift(i,fill_value=0)
        X_final['y'] =  X_bis[f'lagY_{i}'].shift(i, fill_value=0)
        X_final['v'] =  X_bis[f'lagV_{i}'].shift(i,fill_value=0)

    X_final.drop(index=[0,1],axis=0)

    return X_final
