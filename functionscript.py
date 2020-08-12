from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                   # for beautiful plots
from scipy import stats

def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_indexes = []
    kf = KFold(n_splits=4, shuffle=True, random_state=6785)
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y)
        tr_pred = model.predict(tr_x)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_indexes.append(va_idx)
        print('  score Train : {:.6f}' .format(np.sqrt(mean_squared_error(tr_y, tr_pred))),
              '  score Valid : {:.6f}' .format(np.sqrt(mean_squared_error(va_y, pred))))
    # バリデーションデータに対する予測値を連結し、その後元の順番に並べなおす
    va_indexes = np.concatenate(va_indexes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_indexes)
    pred_train = pd.DataFrame(preds[order])
    # テストデータに対する予測値の平均をとる
    preds_test = pd.DataFrame(np.mean(preds_test, axis=0))
    print('Score : {:.6f}' .format(np.sqrt(mean_squared_error(train_y, pred_train))))
    return pred_train, preds_test, model

def missing_check(data,head_count=5):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent],axis=1, keys=['Total','Percent'])
    print('Number of missing columns:', len(missing_data[missing_data['Percent']>0]))
    return missing_data
