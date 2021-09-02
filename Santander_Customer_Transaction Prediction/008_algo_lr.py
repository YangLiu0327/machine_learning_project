import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data(train_data_path, train_label_path, val_data_path, val_label_path):
    val_data = pd.read_csv(val_data_path)
    val_data.drop(columns=['Unnamed: 0'], inplace=True)
    val_label = pd.read_csv(val_label_path)
    val_label.drop(columns=['Unnamed: 0'], inplace=True)
    train_data = pd.read_csv(train_data_path)
    train_data.drop(columns=['Unnamed: 0'], inplace=True)
    train_data.columns = val_data.columns
    train_label = pd.read_csv(train_label_path)
    train_label.drop(columns=['Unnamed: 0'], inplace=True)
    train_label.columns = val_label.columns
    print("Train: {},{}".format(train_data.shape, train_label.shape))
    print("Val: {}, {}".format(val_data.shape, val_label.shape))
    return train_data, train_label, val_data, val_label
    
    
def single_test(model, train_data, train_label, val_data, val_label, head_string, log_path=None):
    train_label = np.ravel(train_label)
    val_label = np.ravel(val_label)
    # fit model
    model.fit(train_data, train_label)
    # get train prediction
    pred_tr = model.predict(train_data)
    pred_tr_proba = model.predict_proba(train_data)[:,1]
    # get cv prediction
    pred_cv = model.predict(val_data)
    pred_cv_proba = model.predict_proba(val_data)[:,1]
    # get metrics
    a_tr = roc_auc_score(train_label, pred_tr_proba)
    a_cv = roc_auc_score(val_label, pred_cv_proba)
    # build progress
    total_time = (time() - start_time) / 60.
    progress = "{}|auroc_tr {:.4f}|auroc_cv {:.4f}"
    progress = progress.format(head_string, a_tr, a_cv)
    print(progress)
    if log_path is not None:
        with open(log_path,'a') as f:
            f.write(progress+"\n")
            
            
if __name__ == "__main__":
    train_data_path = "data/train_data_aug.csv"
    train_label_path = "data/train_label_aug.csv"
    val_data_path = "data/val_data.csv"
    val_label_path = "data/val_label.csv"
    train_data, train_label, val_data, val_label = load_data(train_data_path, train_label_path, val_data_path, val_label_path)
    train_label = train_label.astype('int64')
    
    print("Start running LR")
    # parameter C, helpful to handle overfitting
    # it seems that the result unchanged, thus use default 1.0
    param_grid = [1.0,0.1,0.01,0.001] # default 1.0
    start_time = time()
    for para in param_grid:
        head_string = "C {:<10}".format(para)
        log_path = "result/lg_001_c.txt"
        estimator = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=para)
        single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
        curr_time = (time() - start_time) / 60.
        print("Time: {:.2f}min".format(curr_time))
    
    
    # for para in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
    #     head_string = "solver {:<10}".format(para)
    #     estimator = LogisticRegression(random_state=0, solver=para, max_iter=10000)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string)


