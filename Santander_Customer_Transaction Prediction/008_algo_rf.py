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
    
    print("Start running RF")
    
    # round 1 num_estimators, due to long time, I use 300
    
    # round 2 max_features ---> 'log2'
    # here I use: 'log2'
    # start_time = time()
    # param_grid = ['log2','auto']
    # for para in param_grid:
    #     head_string = "max_features {}".format(para)
    #     log_path = "result/rf_002_max_features.txt"
    #     estimator = RandomForestClassifier(n_estimators=300, 
    #                                         random_state=0, 
    #                                         max_features=para, 
    #                                         n_jobs=-1)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
    #     curr_time = (time() - start_time) / 60.
    #     print("Time: {:.2f}min".format(curr_time))
        
    
    # round 3: max_depth ---> best choice is 60
    # start_time = time()
    # param_grid = [None] + list(range(10,151,10))
    # for para in param_grid:
    #     head_string = "max_depth {}".format(para)
    #     log_path = "result/rf_003_max_depth.txt"
    #     estimator = RandomForestClassifier(n_estimators=300, 
    #                                         random_state=0, 
    #                                         max_features='log2', 
    #                                         max_depth=para,
    #                                         n_jobs=-1)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
    #     curr_time = (time() - start_time) / 60.
    #     print("Time: {:.2f}min".format(curr_time))

    # round 4: min_samples_split ---> best choice is 25
    # start_time = time()
    # param_grid = [2] + list(range(5,151,5))
    # for para in param_grid:
    #     head_string = "min_samples_split {}".format(para)
    #     log_path = "result/rf_004_min_samples_split.txt"
    #     estimator = RandomForestClassifier(n_estimators=300, 
    #                                         random_state=0, 
    #                                         max_features='log2', 
    #                                         max_depth=60,
    #                                         min_samples_split=para,
    #                                         n_jobs=-1)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
    #     curr_time = (time() - start_time) / 60.
    #     print("Time: {:.2f}min".format(curr_time))

    # round 5: min_samples_leaf --> 1 is the best
    # start_time = time()
    # param_grid = [1] + list(range(5,101,5))
    # for para in param_grid:
    #     head_string = "min_samples_leaf {}".format(para)
    #     log_path = "result/rf_005_min_samples_leaf.txt"
    #     estimator = RandomForestClassifier(n_estimators=300, 
    #                                         random_state=0, 
    #                                         max_features='log2', 
    #                                         max_depth=60,
    #                                         min_samples_split=25,
    #                                         min_samples_leaf=para,
    #                                         n_jobs=-1)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
    #     curr_time = (time() - start_time) / 60.
    #     print("Time: {:.2f}min".format(curr_time))
    
    # round 6: max_leaf_nodes --> Use default None
    # start_time = time()
    # param_grid = list(range(120,301,10))
    # for para in param_grid:
    #     head_string = "max_leaf_nodes {}".format(para)
    #     log_path = "result/rf_006_max_leaf_nodes.txt"
    #     estimator = RandomForestClassifier(n_estimators=300, 
    #                                         random_state=0, 
    #                                         max_features='log2', 
    #                                         max_depth=60,
    #                                         min_samples_split=25,
    #                                         min_samples_leaf=1,
    #                                         max_leaf_nodes=para,
    #                                         n_jobs=-1)
    #     single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
    #     curr_time = (time() - start_time) / 60.
    #     print("Time: {:.2f}min".format(curr_time))
    
    # round 7: try to use more estimators again
    start_time = time()
    param_grid = list(range(300,1501,25))
    for para in param_grid:
        head_string = "n_est {}".format(para)
        log_path = "result/rf_007_n_estimator_again.txt"
        estimator = RandomForestClassifier(n_estimators=para, 
                                            random_state=0, 
                                            max_features='log2', 
                                            max_depth=60,
                                            min_samples_split=25,
                                            min_samples_leaf=1,
                                            n_jobs=-1)
        single_test(estimator, train_data, train_label, val_data, val_label, head_string, log_path)
        curr_time = (time() - start_time) / 60.
        print("Time: {:.2f}min".format(curr_time))
