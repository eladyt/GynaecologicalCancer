import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
sns.set_style("white")

# Definitions
path = '/disks/sdc/eyomtov/data_0310/'
REMOVE_DUPLICATE_QUERIES = True
MIN_DAYS_ACTIVE = 90

# Read data
queries2 = pd.read_pickle(path + 'formatted_data/queries2_highres.pkl')
aggregates = pd.read_pickle(path + 'formatted_data/aggregates_highres.pkl')

# Keep only data from before the doctor visit
mask = queries2['dt'] < 0.0

# Keep data from users with more than N days of data
temp = pd.merge(queries2[queries2['dt'] < 0.0][['ID', 'ts']], aggregates[['patientID', 'first_ts']], left_on='ID', right_on='patientID')
temp['active_dt'] = (temp['ts'] - temp['first_ts']).dt.days
s_agg = temp[['patientID', 'active_dt']].groupby(['patientID'], as_index=False).agg({'active_dt' : ['max']}).droplevel(axis=1, level=0).reset_index()
s_agg.columns = ['index', 'patientID', 'active_span']
print("Keeping " + str(len(s_agg[s_agg['active_span'] >= MIN_DAYS_ACTIVE])) + " users with at least " + str(MIN_DAYS_ACTIVE) + " days of data")
keep_users = s_agg[s_agg['active_span'] >= MIN_DAYS_ACTIVE]['patientID'].unique()
mask = mask & (queries2['ID'].isin(keep_users))

# Remove repeating queries
c = 0
lastID = ''
lastQuery = ''
if REMOVE_DUPLICATE_QUERIES:
    for index, row in queries2.iterrows():
        if index > 0:
            if not ((row['patientID'] != lastID) | (row['Query'] != lastQuery)):
                mask[index] = False
                c += 1

        lastID = row['patientID']
        lastQuery = row['Query']

print("There are " + str(len(queries2['ID'].unique())) + " users before removal")

print("Removing " + str((~mask).sum()) + " rows, of those, " + str(c) + " are repeated queries")
queries2 = queries2[mask].reset_index()

print("There are " + str(len(queries2)) + " queries")
print("There are " + str(len(queries2['patientID'].unique())) + " users")

#Some formatting...
queries2['B/M'] = queries2['B/M'].str.strip()
queries2['Label'] = (queries2['B/M'] == 'M-S')

print("Patient classes:")
print(queries2['B/M'].unique())

# Exclude some queries
exclude = queries2['Query'].str.contains(r"\b(?:hunger crossword|nouri appetite|cat|dog|puppy|gum|blood pressure|bloody mary|hypertension|stainless|blood rubies|paint|ankle|knee|finger|rolling stones|tonsil|heel|shoulder|avebury|painshill|bloodline|sign in|sign up|covid|corona|ibstock|turtle|john stones|edging stones|stones to pounds|kilos to stones|kilos in stones|kg to stones|stones to kg)", flags=re.IGNORECASE)

# Try to use the classified keywords
keywords = pd.read_excel(path + 'Classified keywords.xlsx')

for index, kw in keywords.iterrows():
    match_phrase = r"\b(?:" + kw['KWs'] + ")"
    queries2[kw['Category']] = queries2['Query'].str.contains(match_phrase, flags=re.IGNORECASE)
    print(kw['Category'] + ": " + str(queries2[kw['Category']].sum()))

# Plot classes over time per category
for index, kw in keywords.iterrows():
    cur_data = queries2[(queries2[kw['Category']] & ~exclude)].copy()
    cur_data = cur_data.reset_index()

    # Prepare the plot
    cur_data['sampledTime'] = (cur_data['dt'] / 7.0).apply(np.floor)  # In weeks

    gr = cur_data[['Label', 'sampledTime', 'patientID']].groupby(['Label', 'sampledTime'], as_index=False).agg(
        {'sampledTime': [('time', 'mean'), ('numSamples', 'count')], 'patientID': [('userCount', pd.Series.nunique)]}).reset_index()
    gr.columns = gr.columns.get_level_values(0) + '_' + gr.columns.get_level_values(1)

    #Count queries per user
    gr = gr.rename(columns={'Label_' : 'Label', 'sampledTime_time' : 'Time', 'sampledTime_numSamples' : 'numSamples'})

    # Normalize by the total number of users in each class
    n1 = len(queries2['patientID'][queries2['Label']].unique())
    n0 = len(queries2['patientID'][~queries2['Label']].unique())

    gr.loc[gr['Label'] == True, 'numSamples'] = gr.loc[gr['Label'] == True, 'numSamples'] / n1
    gr.loc[gr['Label'] == False, 'numSamples'] = gr.loc[gr['Label'] == False, 'numSamples'] / n0

    # Plot
    plt.figure(index)
    plt.plot(gr[gr['Label']==True]['Time'],  gr[gr['Label']==True]['numSamples'], label='Malignant')
    plt.plot(gr[gr['Label']==False]['Time'], gr[gr['Label']==False]['numSamples'], label='Non-malignant')
    plt.xlabel('Weeks')
    plt.legend(loc='best')
    plt.title(kw['Category'])
    plt.xlim(-52, 0)
    plt.savefig(path + "Charts/" + kw['Category'] + '_52w.jpg')


# Save the queries to file for the classifier
keep_cols = ['Label', 'patientID', 'dt', 'presentation_ts', 'first_ts']
for index, kw in keywords.iterrows():
    keep_cols.append(kw['Category'])
data_for_learning = queries2[queries2.columns.intersection(keep_cols)]
data_for_learning = pd.merge(data_for_learning, aggregates[['patientID', 'first_ts']], left_on='patientID', right_on='patientID')
data_for_learning.to_pickle(path + 'formatted_data/data_for_learning_highres.pkl')
