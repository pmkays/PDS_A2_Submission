from utilities import get_combined_dataset_testset
from utilities import calculate_ndcg
from utilities import export_runfile
import A2
from bs4 import BeautifulSoup
from bs4.element import Comment
from tqdm import trange, tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit
from xgboost.sklearn import XGBRanker
import pandas as pd
import numpy as np
import math
from collections import Counter

dataset, testset, combined_dataset, combined_testset = get_combined_dataset_testset()

#https://medium.com/predictly-on-tech/learning-to-rank-using-xgboost-83de0166229d
def get_train_test_indicies_hillclimb(data): 
    gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state = 27).split(combined_dataset, groups=combined_dataset['QueryID'])
    X_train_inds, X_test_inds = next(gss)
    
    if data == "train":
        return X_train_inds
    else:
       return X_test_inds 
     

def get_train_data_hillclimb(): 
    X_train_inds = get_train_test_indicies_hillclimb("train")
    train_data= combined_dataset.iloc[X_train_inds]

    X_train = train_data.loc[:, ~train_data.columns.isin(['Docid', 'Label', 'QueryID'])]
    y_train = train_data.loc[:, train_data.columns.isin(['Label'])]
    groups = train_data.groupby('QueryID').size().to_frame('size')['size'].to_numpy()

    return train_data, X_train, y_train, groups


def get_test_data_hillclimb(): 
    X_test_inds = get_train_test_indicies_hillclimb("test")
    test_data= combined_dataset.iloc[X_test_inds]

    X_test = test_data.loc[:, ~test_data.columns.isin(['Docid', 'Label', 'QueryID'])]
    y_test = test_data.loc[:, test_data.columns.isin(['Label'])]
    groups_test = test_data.groupby('QueryID').size().to_frame('size')['size'].to_numpy()

    return test_data, X_test, y_test, groups_test


def feature_selection(seed, model, dropped_labels, col_num):
    max_score = 0.0
    new_features = []
    new_indexes = [] 
    randomIndexes=shuffle(range(0,col_num),random_state=seed)
    
    for current_feature in tqdm(range(0,col_num), desc='2nd loop'):
        new_indexes.append(randomIndexes[current_feature])    

        #prepare the data based on the new indexes
        train_data, X_train, y_train, groups = get_train_data_hillclimb() 
        test_data, X_test, y_test, groups_test = get_test_data_hillclimb()

        #extract what's relevant depending on the new indexes and ensure we have a fully labelled dataset for predictions and grouping
        X_train=X_train.iloc[:,new_indexes]  
        X_train_labelled = X_train.join(train_data.loc[:, train_data.columns.isin(dropped_labels)])
        X_test = X_test.iloc[:,new_indexes]

        #train and predict the model
        model.fit(X_train, y_train, group=groups, verbose=False, eval_set=[(X_test, y_test)], eval_group=[groups_test], early_stopping_rounds=100)
        queryIDs, docIDs, predictions = A2.predict_model(model, X_train_labelled, dropped_labels)
        export_runfile(queryIDs, docIDs, predictions, 'hillclimb.csv')

        #determine if the score is higher with the new feature or if it's lower
        current_score=float(calculate_ndcg('hillclimb.csv'))      
        if(current_score <  max_score):
            new_indexes.remove(randomIndexes[current_feature])
        else:
            max_score = current_score
            new_features = X_train.columns

    return  max_score, new_indexes, new_features


def get_features():
  print("Performing feature selection...Check hillclimb-new.txt")
  f = open("hillclimb-new.txt","a")
  col_num=combined_dataset.shape[1]-3 #no queryid, docid, and label

  params={'gamma': 3.62625949981358, 'learning_rate': 0.2691854123225587, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 109} #hyperopt ran on 06/10, s3782041-2/3.tsv
  model = XGBRanker(**params)
  dropped_labels= ['QueryID', 'Label', 'Docid']
  
  #try multiple times and use the iteration number as the seed so that this is reproducable
  for seed in trange(25, desc='1st loop'): 
      max_score, new_indexes, new_features = feature_selection(seed, model, dropped_labels, col_num)
      removed_features= [column for column in combined_dataset.columns if column not in new_features]
      f.write("---------------------------------------------\n")
      f.write(f"MAX SCORE FOR ITERATION {seed}: " + str(max_score) + "\n")
      f.write("New feature indexes: " + str(new_indexes) + "\n")
      f.write("New feature names: " + str(new_features) + "\n")
      f.write("Removed feature names (dropped_labels): " + str(removed_features) + "\n")
      f.write("---------------------------------------------" +"\n")
      f.write("\n")

  print("write complete")
  f.close()







#https://stackoverflow.com/questions/43419803/information-theoretic-measure-entropy-calculation
def calculate_entropy(arr):
    words = Counter(arr)
    total = sum(words.values()) 
    
    #calculate frequencies of each word after getting the counts for each unique word
    pkvec = [value/total for key, value in words.items()]
    
    #calculate entropy
    H = -sum([pk  * math.log(pk) / math.log(2) for pk in pkvec ])
    return H


# https://stackoverflow.com/a/1983219
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta']:
        return False
    if isinstance(element, Comment):
        return False
    return True


#cleans the text from the html column, makes a list of words
def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    cleaned_text = u" ".join(t.strip() for t in visible_texts)
    arr = []
    for x in cleaned_text.strip().split(' '):
        arr.append(x.strip())

    cleaned_no_empty= list(filter(lambda a: a != '', arr))
    return cleaned_no_empty


#obtained from paper linked in week 9
def calculate_fracstop_coverstop(stoplist, html_arr):
    stopword_list_doc=[]
    stopword_count = 0

    for word in html_arr:
        if word in stoplist:
            stopword_count+=1
            stopword_list_doc.append(word)

    nonstopword_count = len(html_arr)-stopword_count
    frac_stop = 0 if nonstopword_count == 0 else stopword_count/nonstopword_count 

    unique_stopword_list_doc= set(stopword_list_doc)
    cover_stop = len(unique_stopword_list_doc)/len(stoplist)

    return frac_stop, cover_stop


def get_stoplist():
  f = open("stoplist_nltk.txt","r")
  stoplist= list(f.read().split("\n"))
  return stoplist


def get_df_fe():
  print("Opening documents.tsv... this may take a while...")
  names=["Docid","Withhtml","Withouthtml"]
  df = pd.read_csv("documents.tsv", header=None, names=names, sep='\t')
  df_fe= testset.merge(df, on="Docid", how="left")
  return df_fe


def export_feature_engineering(df_fe):
  #export the train.tsv/test.tsv with the new columns added in 
  to_export = df_fe.loc[:, ~df_fe.columns.isin(['Withhtml', 'Withouthtml'])]
  to_export.to_csv("fe_complete.tsv", sep='\t', header=True, index=False)
  #export only the new columns which can be appended when reading in train.tsv and test.tsv
  feature_engineered_columns = to_export.loc[:, to_export.columns.isin(['Docid', 'frac_stop','cover_stop','entropy'])]
  feature_engineered_columns.to_csv("fe_cols.tsv", sep='\t', header=True, index=False)

def feature_engineering():
  stoplist = get_stoplist()
  df_fe = get_df_fe() 

  #calculate frac_stop, cover_stop, entropy for each row
  frac_stops, cover_stops, entropies = [],[],[]
  for index, row in tqdm(df_fe.iterrows(), total=df_fe.shape[0]):
      cleaned_no_empty = text_from_html(row["Withhtml"])
      frac_stop, cover_stop = calculate_fracstop_coverstop(stoplist, cleaned_no_empty)
      entropy = calculate_entropy(cleaned_no_empty)
      
      frac_stops.append(frac_stop)
      cover_stops.append(cover_stop)
      entropies.append(entropy)

  #append to the dataframe for exporting
  df_fe['frac_stop'], df_fe['cover_stop'], df_fe['entropy'] = [frac_stops, cover_stops, entropies]
  
  export_feature_engineering(df_fe)
