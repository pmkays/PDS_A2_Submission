FILES IN THE DIRECTORY AND PURPOSE
-----------------------------------------------------------------------
A2.run: tab delimited, 3 columns of queryid, docid and label. Used to score the model.
A2.pdf: report.

A2.py: train the model and predict using the train.tsv, test.tsv, train_feature_engineering_nltk.tsv, and test_feature_engineering_nltk.tsv files.
utilities.py: contains trec_evaluator ndcg scorer and an exporting functino.
feature_engineering_selection.py: contains feature engineering (frac_stop, cover_stop, entropy) and feature selection through hillclimbing as obtained from tutorial 6 (week 7). 
cv.hyperopt.py: contains a group k-folds cross validation and hyperparameter tuning through hyperopt. This hyperopt implementation automatically uses cross validation.

stoplist_nltk.txt: a list of stop words as obtained through the nltk study. 
requirements.txt: for freezing requirements.
hillclimb.txt: an example of feature selection.

test_feature_engineering_nltk.tsv: frac_stop, cover_stop and entropy calculations for train.tsv. This will be combined with train.tsv when reading in the data automatically.
train_feature_engineering_nltk.tsv: frac_stop, cover_stop and entropy calculations for test.tsv. This will be combined with test.tsv when reading in the data automatically.


RUNNING THE PROGRAM
-----------------------------------------------------------------------
"python A2.py" - train the model and predict. 
"python A2.py cv" - performs group k-folds cross validation on some hardcoded parameter values and featuress. (k=5)
"python A2.py params" - performs hyperparameter tuning using hyperopt. Default is 100 evaluations and uses group k-folds cross validation.
"python A2.py fs" - performs feature selection through hillclimbing. Appends to a hillclimb.txt per iteration. 
"python A2.py fe" - uses train.tsv and test.tsv to calculate frac_stop, cover_stop, and entropy. Must have a documents.tsv. Currently hardcoded for test.tsv, can change to train.tsv if desired.
