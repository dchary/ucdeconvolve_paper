# Import base packages required
import sys, os, scipy, random, gc, warnings, re, anndata
import numpy as np
import pandas as pd
import scanpy as sc
from importlib import reload
import unicell as uc
from tqdm.auto import tqdm
import tensorflow as tf

import tempfile
import dask.array as da
import dask.dataframe as dd
import scipy
from glob import glob
from progressbar import progressbar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unicell.tensorflow.deconvolve.deconvutils import r2_keras, correlation, soft_acc, sparseMSE, ccc_mask, ccc_full
from unicell.tensorflow.deconvolve.deconvutils import r2_keras_primary, r2_keras_cell_lines, r2_keras_cancer_subtypes


# Set google credentials for unicell package
uc.settings.set_credentials(creds_gsuite_path = os.environ['GSUITE_CREDS'],
                            creds_gcp_path = os.environ['GCP_CREDS'])
uc.settings.set_project_id(os.environ['PROJECT_ID'])
uc.settings.set_gcs_bucket(os.environ['BUCKET'])


# Get a strategy
strategy = tf.distribute.MirroredStrategy()

# Base data
TEST_TRAIN_RATIO = 0.2
N_EXAMPLES = 10000000 
BATCH_SIZE = 256 * 8 # Optimal size to distribute to each core / device
SHUFFLE_BUFFER_SIZE = 10000
N_EPOCHS = 50
N_WORKERS = 96
VALIDATION_FREQ = 5

# Collect training data
training_data_path = os.environ['TRAINING_DATA_PATH']
train_dataset, test_dataset = uc.tf.deconvolve.create_deconvolve_training_data(training_data_path, 
                                                                               shuffle_buffer_size = SHUFFLE_BUFFER_SIZE, 
                                                                               batch_size = BATCH_SIZE)

# Generate steps per epoch for train and test
STEPS_PER_EPOCH = int(np.floor((1-TEST_TRAIN_RATIO) * (N_EXAMPLES / (BATCH_SIZE))))
VAL_STEPS_PER_EPOCH = int(np.floor(TEST_TRAIN_RATIO * (N_EXAMPLES / (BATCH_SIZE))))

# Save path for checkpoints
CHECKPOINTS_PATH = os.environ['CHECKPOINTS_PATH']

# Flag to restore from last good model and define checkpoint path on where to save
REFRESH_WEIGHTS = False

# Add nothing

# Use strategy context to build and compile model
with strategy.scope():
    
    # Get model
    model = uc.tf.deconvolve.models.get_model_mk1()

    # Compile with optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001) # Default = 0.00001
    model.compile(optimizer = opt, loss = sparseMSE, 
                  metrics=[r2_keras, ccc_full, ccc_mask], run_eagerly = None)

    # Refresh weights
    if REFRESH_WEIGHTS: model.load_weights(CHECKPOINTS_PATH)
    
    # Get model summary
    model.summary()
    
    model_mem_usage = uc.tf.utils.keras_model_memory_usage_in_bytes(model, batch_size = BATCH_SIZE, human_readable = True)
    print(f"Estimted Memory Usage: {model_mem_usage}")
    
# Ignore warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

# Define a checkpoint structure to save our model during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH, monitor='loss', verbose=1, save_best_only = True, save_weights_only = True, mode='auto', period=1)
earlystop = tf.keras.callbacks.EarlyStopping(patience = 4, monitor = 'loss', min_delta= 0.0000125, verbose = 1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 5, verbose = 1)
slackNotify = uc.tl.SlackNotifyCallBack()

callbacks = [checkpoint, reduce_lr, slackNotify]

# Actually do the training
model.fit(train_dataset, validation_data = test_dataset, epochs = N_EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH, use_multiprocessing = True,  verbose = 1,
    workers = N_WORKERS, validation_freq = VALIDATION_FREQ, validation_steps = VAL_STEPS_PER_EPOCH, 
    class_weight = None, callbacks = callbacks)