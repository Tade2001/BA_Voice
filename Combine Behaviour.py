from os import listdir
import json
import pandas as pd
import pathlib
import os
import matplotlib
matplotlib.use('TkAgg')
import mne

# Set directories
DIR = pathlib.Path(os.getcwd())
behavior_results_DIR = DIR
with open(DIR / "preproc_config.json") as file:
    cfg = json.load(file)


### Analysis within participants ###
subj= "sub_28" # Define which participant's data to analyze


# Define directories for subject
EEG_DIR = DIR / "Results" / subj / "evokeds"
evoked = mne.read_evokeds(EEG_DIR / pathlib.Path(subj+'-ave.fif'))
behavior_dir= DIR / "Data" / subj
plots_DIR= DIR / "Results" / subj
if not os.path.isdir(plots_DIR):
        os.makedirs(plots_DIR)


# Summarize and plot the behavioral data WICHTIG
files = [file for file in listdir(behavior_dir) if file.endswith('.csv')]
files = [k for k in files if 'experiment' in k]
data = pd.read_csv(str(behavior_dir) + '/' + files[0])

morphs= list(data['Morph played'].unique())
results_dict=dict()
i=0
for morph in morphs:
    df_sub = data[data["Morph played"] == morph]
    response_count = list(df_sub['Response'].value_counts().values)
    responses = list(df_sub['Response'].value_counts().index.values)
    if len(response_count) != 1:
        no = response_count[responses.index(2)]
        yes = response_count[responses.index(1)]
    elif 2 not in responses:
        no = 0
        yes = response_count[responses.index(1)]
    elif 1 not in responses:
        no = response_count[responses.index(2)]
        yes = 0
    results_dict[str(i)] = {"Morph ratio": morph, "Voice responses": yes, "N": sum(response_count) }
    i = i + 1

results=pd.DataFrame.from_dict(results_dict, orient='index')
results["%Voice"] = (results["Voice responses"] / results["N"])
results=results.sort_values(by=['Morph ratio']).sort_values(by=['Morph ratio'])
results['Morph ratio'] = results['Morph ratio'].astype(float)
results = results.assign(subj=[subj] * len(morphs))
results.to_csv(str(behavior_results_DIR) +'/'+ subj + '_summarized-behavior.csv')
