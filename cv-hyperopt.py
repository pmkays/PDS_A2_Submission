from utilities import get_combined_dataset_testset
from utilities import calculate_ndcg
from utilities import export_runfile
from A2 import predict_model
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import pandas as pd
import numpy as np


dataset, testset, combined_dataset, combined_testset = get_combined_dataset_testset()


def get_x_y_cv(train_index, test_index, X, y, dropped_labels):
    #prepare the data for train/test and return everything we need
    X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
    X_train, X_test = X_train_raw.loc[:, ~X_train_raw.columns.isin(dropped_labels)], X_test_raw.loc[:, ~X_test_raw.columns.isin(dropped_labels)]  
    y_train, y_test = y[train_index], y[test_index]

    groups = X_train_raw.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
    groups_test=  X_test_raw.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
    
    return X_train_raw, X_train, y_train, X_test_raw, X_test, y_test, groups, groups_test


def perform_cv(model, dropped_labels):
    X = dataset
    y = dataset['Label']
    groupsforcv = dataset['QueryID'];
    group_kfold = GroupKFold(n_splits=5)
    scores=[]
    
    fold = 0
    for train_index, test_index in group_kfold.split(X, y, groupsforcv):
        fold+=1
        
        #need a copy of the raw data so we can still group by queryid
        X_train_raw, X_train, y_train, X_test_raw, X_test, y_test, groups, groups_test = get_x_y_cv(train_index, test_index, X, y, dropped_labels)

        model.fit(X_train, y_train, group=groups, verbose=False, eval_set=[(X_test, y_test)], eval_group=[groups_test], early_stopping_rounds=100)
        queryIDs, docIDs, predictions = predict_model(model, X_test_raw, dropped_labels)
        export_runfile(queryIDs, docIDs, predictions, f'f{str(fold)}.csv')
        
        score=calculate_ndcg(f'f{str(fold)}.csv')      
        scores.append(float(score))
        print("SCORE FOR FOLD " + str(fold) + ": " + str(score))
            
    return sum(scores)/len(scores)

def cv():
  dropped_labels=['QueryID', 'AnchorTerms', 'TitleTerms', 'URLTerms', 'TFIDFBody', 'TFIDFAnchor', 'TFIDFTitle', 'TFIDFURL', 'TFIDFWholeDocument', 'LengthAnchor', 'LengthTitle', 'LengthWholeDocument', 'BM25Anchor', 'BM25Title', 'BM25URL', 'BM25WholeDocument', 'LMIRABSTitle', 'LMIRDIRAnchor', 'LMIRDIRURL', 'LMIRDIRWholeDocument', 'LMIRIMBody', 'LMIRIMURL', 'LMIRIMWholeDocument', 'PageRank', 'InlinkNum', 'NumSlashURL', 'LenURL', 'NumChildPages', 'Docid', 'Label']
  # params ={'gamma': 4.3331098918759485, 'learning_rate': 0.2913689488086962, 'max_depth': 2, 'min_child_weight': 6, 'n_estimators': 376} 
  params={'gamma': 3.62625949981358, 'learning_rate': 0.2691854123225587, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 109} #hyperopt ran on 06/10
  model = xgb.sklearn.XGBRanker(**params)
  average_ndcg = perform_cv(model, dropped_labels)
  print ("AVERAGE SCORE:", average_ndcg)





space={"n_estimators": hp.quniform('n_estimators',50,500,1),
        'gamma': hp.uniform('gamma', 1,10),
        "max_depth": hp.quniform('max_depth',1,10,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        "learning_rate": hp.uniform('learning_rate',0,2),
        'seed': 27 }

def objective(space):
    model=xgb.sklearn.XGBRanker(
                    n_estimators =int(space['n_estimators']), gamma = space['gamma'], max_depth = int(space['max_depth']), 
                    min_child_weight=int(space['min_child_weight']), learning_rate = space['learning_rate']) 
    
    dropped_labels=['QueryID', 'AnchorTerms', 'TitleTerms', 'URLTerms', 'TFIDFBody', 'TFIDFAnchor', 'TFIDFTitle', 'TFIDFURL', 'TFIDFWholeDocument', 'LengthAnchor', 'LengthTitle', 'LengthWholeDocument', 'BM25Anchor', 'BM25Title', 'BM25URL', 'BM25WholeDocument', 'LMIRABSTitle', 'LMIRDIRAnchor', 'LMIRDIRURL', 'LMIRDIRWholeDocument', 'LMIRIMBody', 'LMIRIMURL', 'LMIRIMWholeDocument', 'PageRank', 'InlinkNum', 'NumSlashURL', 'LenURL', 'NumChildPages', 'Docid', 'Label']
    average_ndcg = perform_cv(model, dropped_labels)
    
    print ("SCORE:", average_ndcg)
    return {'loss': -average_ndcg, 'status': STATUS_OK }

def hyperparameter_tuning():
  trials = Trials()

  best_hyperparams = fmin(fn = objective,
                          space = space,
                          algo = tpe.suggest,
                          max_evals = 100,
                          trials = trials)

  print("The best hyperparameters are : ","\n")
  print(best_hyperparams)
  return best_hyperparams