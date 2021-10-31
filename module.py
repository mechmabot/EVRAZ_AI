import numpy as np
import pandas as pd
#FEATURES 
def make_feature_changing(df, feature_changing_list):
    for func, params in feature_changing_list:
        df = func(df=df, **params)
    return df


#MULTIPLE PREDICTIONS
def multiple_predictions_TST(models_TST, Xs_TST):
    predicts = []
    for idx, model in enumerate(models_TST):
        predict = model.predict(Xs_TST[idx])
        predicts.append(predict)
    return np.array(predicts).T


def multiple_predictions_C(models_C, Xs_C):
    predicts = []
    for idx, model in enumerate(models_C):
        predict = model.predict(Xs_C[idx])
        predicts.append(predict)
    return np.array(predicts).T


# METRICS
def metric(answers, user_csv):
    delta_c = np.abs(np.array(answers['C']) - np.array(user_csv['C']))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers['TST']) - np.array(user_csv['TST']))
    hit_rate_t = np.int64(delta_t < 20)
    N = np.size(answers['C'])
    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def metric_log(answers, user_csv):
    delta_c = np.abs(np.exp(np.array(answers['log_C'])) - np.exp(np.array(user_csv['log_C'])))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers['TST']) - np.array(user_csv['TST']))
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(answers['log_C'])

    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def metric_C(y_true, y_pred):
    delta_c = np.abs(y_true - y_pred)
    hit_rate_c = np.int64(delta_c < 0.02)
    return hit_rate_c.mean()


def metric_TST(y_true, y_pred):
    delta_t = np.abs(y_true - y_pred)
    hit_rate_t = np.int64(delta_t < 20)

    return hit_rate_t.mean()


def metric_C_log(y_true, y_pred):
    delta_c = np.abs(np.exp(np.array(y_true)) - np.exp(np.array(y_pred)))
    hit_rate_c = np.int64(delta_c < 0.02)

    return hit_rate_c.mean()