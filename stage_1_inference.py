#fundamental libraries
import os
import gc
import re
import math
import logging
import importlib
import numpy as np
import pandas as pd

#utility to read & show progress
import csv
from tqdm.auto import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET

#project specific framework & functions
from sentence_transformers import SentenceTransformer

import utils

#set up paths
root = 'data'
# fine_tuned_model_path = '/content/drive/MyDrive/Datasets/Paraphrase_detection/paraphrase_finetuning_training-paraphrase-multilingual-mpnet-base-v2-2022-07-22_09-20-47'
fine_tuned_model_path = 'models/paraphrase_finetuning_training-paraphrase-multilingual-mpnet-base-v2-2022-07-22_09-20-47'

# print(fine_tuned_model_path)
#get validation dataset sentences
validation_sentences = utils.get_validation_sentences(root)

print("Loading model..........")
#pass them through sentences encoder to get sentence pairs
model = SentenceTransformer(fine_tuned_model_path)

solution, _ = utils.get_solution_parapharse_mining(model,validation_sentences)

del model
gc.collect()


thres = 0.96
key_ind_subset = list(validation_sentences.keys())
soln_pairs = {}
scores_key = {}

#filter pairs having same first id
for (score, id1,id2) in solution:
    if 1.0 > score > thres:
        if key_ind_subset[id1] not in soln_pairs.keys():
            soln_pairs[key_ind_subset[id1]] = [key_ind_subset[id2]]
            scores_key[key_ind_subset[id1]] = [score]
        else:
            soln_pairs[key_ind_subset[id1]].append(key_ind_subset[id2])
            scores_key[key_ind_subset[id1]].append(score) 

#select only pairs with highest similarity score from those pairs
filtered_pairs = {}
for k in soln_pairs.keys():
    filtered_pairs[k] = soln_pairs[k][0]

#if the pair a,b & b,c are both present in the solutions, select the pair with highest similarity
lst = []
new_pairs = {}
for k in filtered_pairs.keys():
    id2 = filtered_pairs[k]
    if id2 in filtered_pairs.keys():
        if 1.0 > scores_key[id2][0] > scores_key[k][0]:
            new_pairs[id2] = filtered_pairs[id2]
            lst.append(id2)
        else:
            new_pairs[k] = id2
    else:
        new_pairs[k] = id2

val_df = pd.DataFrame(new_pairs.items())
val_df.columns = ['id1','id2']
df_name = f"filtered_with_mpnet_v2_results_{thres}.csv"
val_df.to_csv(df_name,index=False)
print("successfully generated predictions for stage 1.......")