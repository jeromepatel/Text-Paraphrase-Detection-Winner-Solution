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
from sentence_transformers import CrossEncoder

import utils

#set up paths
root = 'data'

#get validation dataset sentences
validation_sentences = utils.get_validation_sentences(root)

#threshold for paraphrase detection first stage sparse preds
thres = 0.96
#get pairs from saved intermediate pred file
preds = pd.read_csv(f'filtered_with_mpnet_v2_results_{thres}.csv')

pairs = []
ids = []
for i,(k,v) in preds.iterrows():
    pairs.append((validation_sentences[k], validation_sentences[v]))
    ids.append((k,v))

#generate predictions using cross encoder model
model_name = 'cross-encoder/stsb-roberta-large'
cross_enc = CrossEncoder(model_name)
predictions_confs = cross_enc.predict(pairs)

cross_enc_thres = 0.66

soln_pairs = {}
for i,p in enumerate(predictions_confs):
    #highest similarity is used as id pairs
    if  p > cross_enc_thres and not ids[i][1] in soln_pairs.values():
        soln_pairs[ids[i][0]] = ids[i][1]

print("length after filtering scores is: ",len(soln_pairs))
#if score(id1,id2) > score(id2,id3) include id1,id2 pair else include id2,id3 pair
new_pairs = {}
for k in soln_pairs.keys():
    id1 = k
    id2 = soln_pairs[k]
    #skip if both the sentences are same
    if validation_sentences[id1] != validation_sentences[id2]:
        if id2 in soln_pairs.keys() and predictions_confs[[x[0] for x in ids].index(id2)] > predictions_confs[[x[0] for x in ids].index(k)]:
            new_pairs[id2] = soln_pairs[id2]
        else:
            new_pairs[k] = id2

print(f"percentage retained for {cross_enc_thres} is after filtering: {len(new_pairs)/len(ids)} and final prediction length is: {len(new_pairs)}")

val_df = pd.DataFrame(new_pairs.items())
val_df.columns = ['id1','id2']
df_name = f"cross_enc_result_{cross_enc_thres}.csv"
val_df.to_csv(df_name,index=False)

print("successfully generated predictions for stage 2.......")