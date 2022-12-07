import pandas as pd
import numpy as np
import pickle
from data_generation import *
#ML_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def return_bounce_prediction(xy):

    # Tracknet data
    test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

    X_TS = generate_X10(test_df)
    X_ML = generate_X_ML(test_df)

    # load the pre-trained classifier

    model_TS = pickle.load(open('TSFClassifier2.pkl', "rb"))
    model_ML = pickle.load(open('SVC2.pkl', "rb"))

    # Trying to filter "fake" bounces
    predcted = model_TS.predict(X_TS)

    pred_series = pd.Series(predcted)
    data_with_preds = pd.concat([test_df,pred_series],axis=1)
    data_with_preds.columns=['x', 'y', 'V', 'bounce']
    data_with_preds = data_with_preds.drop(['x', 'V'], axis=1)
    filtered_data_with_pred = filter_fake_bounces(data_with_preds)
    filtered_pred = filtered_data_with_pred['bounce']
    filtered_pred = filtered_pred.to_numpy()

    idx = list(np.where(filtered_pred == 1)[0])
    idx = np.array(idx) - 10

    return idx
