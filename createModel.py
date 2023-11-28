import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sns.set_style("white")
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import plot_importance
from sklearn.model_selection import GroupKFold

# Definitions
path = '/disks/sdc/eyomtov/data_0310/'
first_dt = -180
last_dt = -0
Nframes = 10
non_pattern_cols = ['Label', 'patientID', 'dt', 'index', 'presentation_ts', 'first_ts', 'active_dt', 'active_dt_capped']

mdl = GradientBoostingClassifier(n_estimators=50)

data = pd.read_pickle(path + 'formatted_data/data_for_learning_highres.pkl')
patient_data = pd.read_pickle(path + 'formatted_data/patient_data_highres.pkl')

for col in data.columns:
    if (col not in non_pattern_cols):
        data[col] = data[col].astype(float)

data['active_dt'] = (data['presentation_ts'] - data['first_ts']).dt.days
data['active_dt_capped'] = np.minimum(data['active_dt'], last_dt-first_dt)

mask = (data['dt'] >= first_dt) & (data['dt'] < last_dt)

data = data[mask].reset_index()

# Put the columns into a matrix
queryCount = data[['patientID', 'active_dt_capped']].groupby(['patientID'], as_index=False).count()
queryCount.columns = ['patientID', 'query_count']

gr = data.groupby(['patientID'], as_index=False).sum(numeric_only=True)
gr = pd.merge(gr, queryCount, left_on='patientID', right_on='patientID')
gr['active_dt_capped'] = gr['active_dt_capped'] / gr['query_count'] # This is required because we used sum aggregation on all columns
gr['active_dt'] = gr['active_dt'] / gr['query_count'] # This is required because we used sum aggregation on all columns

for col in gr.columns:
    if (col not in non_pattern_cols):
        gr[col] = gr[col]/gr['active_dt_capped']

gr = gr.drop(columns=['active_dt_capped', 'query_count'])
selected_cols = gr.columns.difference(non_pattern_cols)

patterns = gr[selected_cols].to_numpy()
targets = gr['Label'].to_numpy() > 0

loo = LeaveOneOut()
predictions = cross_val_predict(mdl, patterns, targets, cv=loo, method='predict_proba')

fpr, tpr, _ = roc_curve(targets, predictions[:,1])
roc_auc = auc(fpr, tpr)
print("AUC for model: {:4.2f}".format(roc_auc))
print("Average of labels: {:4.2f}".format(targets.mean()))
print("Number of samples: {:d}".format(len(targets)))

