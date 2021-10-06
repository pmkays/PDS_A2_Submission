import xgboost as xgb
import pandas as pd
import numpy as np
from utilities import get_combined_dataset_testset
from utilities import calculate_ndcg
from utilities import export_runfile

dataset, testset,combined_dataset, combined_testset = get_combined_dataset_testset()


def predict_model(model, X_test_raw, dropped_labels):    
  #these will be used for later exporting
  predictions = []
  queryIDs = []
  docIDs = [] 

  for name, group in X_test_raw.groupby('QueryID'):
      queryIDs = np.append(queryIDs, group['QueryID'])
      docIDs = np.append(docIDs, group['Docid'])
      
      #predictions must happen by group
      droppedGroup = group.drop(dropped_labels,axis=1)
      groupPredictions = model.predict(droppedGroup)
      predictions= np.append(predictions, groupPredictions);
          
  return queryIDs, docIDs,predictions 

def train_model(dropped_labels):
  #set up the data we need, make sure to use only the selected features from hillclimbing
  X_train = combined_dataset.loc[:, ~combined_dataset.columns.isin(dropped_labels)]
  y_train = combined_dataset['Label']
  groups =  combined_dataset.groupby('QueryID').size().to_frame('size')['size'].to_numpy()
  X_test = combined_testset.loc[:, ~combined_testset.columns.isin(dropped_labels[:-1])] #exclude label since it's not in the testset
  
  #params obtained from hyperopt hyperparameter tuning
  params={'gamma': 3.62625949981358, 'learning_rate': 0.2691854123225587, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 109} #hyperopt ran on 06/10
  model = xgb.sklearn.XGBRanker(**params)
  model.fit(X_train, y_train, group=groups, verbose=False) 
  return model


if __name__=="__main__": 
  dropped_labels= ['QueryID', 'AnchorTerms', 'TitleTerms', 'URLTerms', 'TFIDFBody', 'TFIDFAnchor', 'TFIDFTitle', 'TFIDFURL', 'TFIDFWholeDocument', 'LengthAnchor', 'LengthTitle', 'LengthWholeDocument', 'BM25Anchor', 'BM25Title', 'BM25URL', 'BM25WholeDocument', 'LMIRABSTitle', 'LMIRDIRAnchor', 'LMIRDIRURL', 'LMIRDIRWholeDocument', 'LMIRIMBody', 'LMIRIMURL', 'LMIRIMWholeDocument', 'PageRank', 'InlinkNum', 'NumSlashURL', 'LenURL', 'NumChildPages', 'Docid', 'Label']
  model = train_model(dropped_labels)
  queryIDs, docIDs, predictions = predict_model(model, combined_testset, dropped_labels[:-1])
  export_runfile(queryIDs, docIDs, predictions, 'S3782041-1.tsv')