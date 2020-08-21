from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                   # for beautiful plots
from scipy import stats
import math

# plotly系-----------
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
#--------------------------------

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

def object_countplot(data, add_cols,col_num=3,label_vertical=True):
    # データセットの中でobject型の変数をラベルごとに棒グラフにする
    # col_numは表示するグラフの列の数
    # label_verticalはラベルを縦に表示するか横に表示するか決める
    data_object_columns = data.select_dtypes('object').columns
    obj_cols = list(data_object_columns)
    obj_cols = obj_cols + add_cols
    nr_rows = math.ceil(len(obj_cols)/col_num)
    nr_cols = col_num
    subplot_ratio = [4,3]
    if label_vertical:
        subplot_ratio = [4,5]


    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*subplot_ratio[0],nr_rows*subplot_ratio[1]))

    for r in range(0,nr_rows):
        for c in range(0,nr_cols):
            i = r*nr_cols+c
            if i < len(obj_cols):
                g = sns.countplot(x=obj_cols[i], data=data, ax = axs[r][c])
                if label_vertical:
                    a = data[obj_cols[i]].value_counts().index
                    g.set_xticklabels(a, rotation=90)

    plt.tight_layout()
    plt.show()

def object_label(var_name,tar_name, data):
    #変数の中のカテゴリーごとにTargetの0,1の割合を集計
    #tar_nameは0,1ラベルの変数
    #この変数は1の割合を示している
    zentai = data[var_name].value_counts()
    target_1 = data[data[tar_name]==1][var_name].value_counts()
    per = target_1/zentai
    per = per.sort_values(ascending=False)

    #プロットを行う
    labels = list(per.index)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rect = ax.bar(x, per, width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=-90)
    plt.show()
    #データフレームにして返す
    df = pd.concat([per,zentai,target_1],axis=1)
    df.columns = ['tar_per','count','tar1_count']
    return df

def iplt_countplot(var_name,tar_name,data):
    #iplotでカテゴリ変数のカテゴリごとにtargetの数を集計
    temp = data[var_name].value_counts()
    #print(temp.values)
    temp_y0 = []
    temp_y1 = []
    for val in temp.index:
        temp_y1.append(np.sum(data[tar_name][data[var_name]==val] == 1))
        temp_y0.append(np.sum(data[tar_name][data[var_name]==val] == 0))
    trace1 = go.Bar(
        x = temp.index,
        y = (temp_y1 / temp.sum()) * 100,
        name='YES'
    )
    trace2 = go.Bar(
        x = temp.index,
        y = (temp_y0 / temp.sum()) * 100,
        name='NO'
    )

    fig_data = [trace1, trace2]
    layout = go.Layout(
        title = var_name + ' for  '+ tar_name,
        #barmode='stack',
        width = 1000,
        xaxis=dict(
            title=var_name,
            tickfont=dict(
                size=10,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='Count in %',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
    )
    )
    #プロットを行う
    fig = go.Figure(data=fig_data, layout=layout)
    iplot(fig)
