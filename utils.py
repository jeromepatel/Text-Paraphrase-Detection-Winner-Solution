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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample

import nltk
nltk.download('stopwords')
nltk.download('punkt')

#reading info file 
def read_info(file_name):
    #print(file_name)
    xml_data = open(file_name,'r', encoding="utf8").read()
    root = ET.XML(xml_data)
    data = []
    for child in root:
        data.append(subchild.text for subchild in child)
    df = pd.DataFrame(data)
    df.columns = list(s.tag for s in root[0])
    return df

def create_id(file_id,para_id,sent_id):
    file_id = int(re.search(r"\d+", file_id).group(0))
    return f'{file_id}_{para_id}_{sent_id}'

#get dataset info extracted from files inside the root directory
def get_dataset_info(root,training_info):
    #get dataset from list of files
    dataset_info = {}
    cnt_none = 0
    print("loading training info...... ")
    for file in tqdm(training_info):
        df = read_info(os.path.join(root,file))
        #the df will contain file_names in columns if its empty
        if 'file_names' in df.columns:
            #skip that file in training_info
            df = None
            cnt_none += 1
        else:
            df['source_ids'] = df.apply(lambda x: create_id(x.file_ids, x.para_ids,x.sens_ids), axis=1)
            df['target_ids'] = df.apply(lambda x: create_id(x.tag_file_ids, x.tag_para_ids,x.tag_sens_ids), axis=1)
        dataset_info[file.split(".")[0]] = df
    print(f"Total no of non pleg files in dataset is: {(len(training_info)-cnt_none)} and percentage is: {(len(training_info)-cnt_none)/len(training_info)}")
    return dataset_info

#create dataset for fine tuning bi-encoder model
def get_input_examples(dataset):
    sentence_dataset = []
    for key in dataset.keys():
        if dataset[key] is not None:
            for df in dataset[key].itertuples():
                #create a random score between 0.93 & 1.0
                score = np.random.uniform(low=0.93, high=1.0)
                sent = df[5]
                target = df[9]
                input_sample = InputExample(texts=[sent,target], label=score)
                sentence_dataset.append(input_sample)
    return sentence_dataset

#get validation sentences processed from indices files
def get_validation_sentences(root):
    val_sentences = {}
    for filename in tqdm(os.listdir(os.path.join(root,'validation_data/validation_text'))):
        ind_filename = filename.replace('newtext','xml')
        if not filename.endswith('newtext'):
            print(f"{filename} is not a newtext file")
            continue
        doc_number = ind_filename.split('.')[0][7:]
        df = read_info(os.path.join(root,'indices',ind_filename))
        df['indices'] = df['indices'].apply(pd.eval)
        for index, row in df.iterrows():
            sens_id = "_".join([doc_number, str(row['indices'][0]), str(row['indices'][1])])
            # print(doc_number, filename, row['indices'], sens_id)
            val_sentences[sens_id] = row['sentences']
    return val_sentences

#predict pairs by using fine-tuned model to 
def get_solution_parapharse_mining(model,dataset):
    #iterate over dataset to  find the cosine distances
    rev_subset = {}
    for a,b in dataset.items():
        rev_subset[b] = a
    #use inbuilt function for paraphrase mining
    #update the query chunck size to 10k & corpus chunk size to 80k to get most performance (ofc according to your system)
    solution = util.paraphrase_mining(model,list(dataset.values()),query_chunk_size=10000,corpus_chunk_size=20000,show_progress_bar=True, top_k = 10)
    return (solution,rev_subset)