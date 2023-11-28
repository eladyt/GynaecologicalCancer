import pandas as pd
import glob
import os
from scipy.stats import spearmanr
import numpy as np

path = '/disks/sdc/eyomtov/data_0310/'
excel_fn = 'TAKEOUT ANON 03102022.xlsx'

# Read queries files
all_files = glob.glob(path + "TakeoutFilter/*-queries.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    id = int(filename[len(path)+14:len(path)+18])
    df['patientID'] = id
    li.append(df)

queries = pd.concat(li, axis=0, ignore_index=True)
queries['ts'] =  pd.to_datetime(queries['Date'], format='%Y-%m-%d %H:%M:%S +0000')

# Read summaries files
all_files = glob.glob(path + "TakeoutFilter/*-aggregates.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    id = int(filename[len(path)+14:len(path)+18])
    df['patientID'] = id
    li.append(df)

aggregates = pd.concat(li, axis=0, ignore_index=True)
aggregates['first_ts'] =  pd.to_datetime(aggregates['First_Query_Date'], format='%Y-%m-%d %H:%M:%S +0000')

patient_data = pd.read_excel(path + excel_fn)

patient_data['presentation_ts'] = pd.to_datetime(patient_data['DOP GP'], format='%Y-%m-%d %H:%M:%S')

# Join the patient data with the queries data to be able to calculate
# relative time (and have the label)
queries2 = pd.merge(queries, patient_data[['ID', 'presentation_ts', 'AGE', 'B/M']], left_on='patientID', right_on='ID')

queries2['dt'] = (queries2['ts'] - queries2['presentation_ts']).dt.total_seconds() / 86400.0

# Print time spans
print('Queries range from ' + str(queries2['dt'].min()) + ' days to ' + str(queries2['dt'].max()) + ' days')

# Save data
queries.to_pickle(path + 'formatted_data/queries_highres.pkl')
queries2.to_pickle(path + 'formatted_data/queries2_highres.pkl')
patient_data.to_pickle(path + 'formatted_data/patient_data_highres.pkl')
aggregates.to_pickle(path + 'formatted_data/aggregates_highres.pkl')

