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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample

import utils


#the dataset is stored in google drive
root = 'data/training_data/training/info'
training_info = os.listdir(root)
dataset_info = utils.get_dataset_info(root,training_info)

dataset = utils.get_input_examples(dataset_info)

#just using a part of training dataset to showcase fine tuning progress, not important as we are training for 2 epochs only
training_dataset = dataset
validation_dataset = dataset[-100:]


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#fine tuning the model
model_name = 'paraphrase-multilingual-mpnet-base-v2'
train_batch_size = 16
num_epochs = 2
model_save_path = 'models/paraphrase_finetuning_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Setting up dataloader.....")

#create dataloader for training and validation dataset and fine tuning model
train_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Setting up dataloader dev dataset.....")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_dataset, name='para-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

#actual fine tuning
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=200,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


