import pandas as pd
import re
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut
import random
import sys

path = '/disks/sdc/eyomtov/data_0310/'
first_dt = -630
last_dt = 0
sample_data = True
Nsamples = 154

mdl = RandomForestClassifier(n_estimators=100)

# Read the queries file and check which user asked about which terms
questionnaire = pd.read_excel(path + 'Online Searches 24112022 EXPORT.xlsx', sheet_name='Formatted')
keywords = pd.read_excel(path + 'Questionnaire items to query terms.xlsx')

if sample_data:
    users = questionnaire['patientID'].unique().tolist()
    print("Selecting " + str(Nsamples) + " random users")

    sampled_users = random.sample(users, Nsamples)

    questionnaire = questionnaire[questionnaire['patientID'].isin(sampled_users)]
else:
    Nsamples = len(questionnaire)

#Change missing values to -1
for index, kw in keywords.iterrows():
    current_term = kw['Questionnaire item']
    missing_values = ~((questionnaire[current_term]==0.0) | (questionnaire[current_term]==1.0))
    questionnaire.loc[missing_values, current_term] = -1.0

#Can we predict the outcome from the questionnaire response?
qitems = keywords['Questionnaire item'].to_list()
patterns = questionnaire[questionnaire.columns[questionnaire.columns.isin(qitems)]].to_numpy()
questionnaire['Label2'] = questionnaire['Label'].str.strip() == 'M-S'
targets = questionnaire['Label2'].to_numpy()

loo = LeaveOneOut()
predictions = cross_val_predict(mdl, patterns, targets, cv=loo, method='predict_proba')

fpr, tpr, _ = roc_curve(targets, predictions[:,1])
roc_auc = auc(fpr, tpr)

print(str(Nsamples) + "\t" + str(roc_auc) + "\t" + str(targets.mean()) + "\t" + str(len(targets)))

