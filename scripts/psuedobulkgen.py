####################################################################################################
## Copyright (C) 2021-Present - Daniel Charytonowicz - All Rights Reserved
## Unauthorized copying of this file, via any medium is strictly prohibited
## Proprietary and Confidential
## Written by: Daniel Charytonowiczz
## Contact: daniel.charytonowicz@icahn.mssm.edu
####################################################################################################

"""

======
PseudobulkGen: A Pseudobulk RNA sample generation subroutine used to generate sample data for
UniCell Deconvolve.
======


"""

import anndata
from progressbar import progressbar
from numba import jit, njit, prange
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import unicell as uc
from itertools import cycle, repeat
import os
import subprocess

@njit(parallel=True, fastmath=True, error_model='numpy')
def _get_sample_as_mean_exp(B, sample_row, CELL_TYPE_IDXS_ARR, CELL_TYPE_IDXS_ARR_INDPTR, X_INDPTR, X_INDICES, X_DATA):
    """
    New utility to quickly get rows of large matrix from a block of random rows.
    
    :param B: the result buffer to place result in, must be a ndarray of size (28867,)
    :param sample_row: a sample row that comprises the index of the cell type we need and how many cells of that type
    :param CELL_TYPE_IDXS_ARR: cell_type index array telling us where to get information from.
    :param CELL_TYPE_IDXS_ARR_INDPTR: index pointer to cell type index array for fast slicing
    :param X_INDPTR: indptr for X_INDICES and X_DATA for fast slicing.
    :param X_INDICES: column indices for X_DATA
    :parma X_DATA: data from database as CSR data record.
    
    """
    
    # Use argwhere to collect the cell type based on the column index
    cell_type_idxs = np.argwhere(sample_row).flatten()
    cell_type_counts = sample_row[cell_type_idxs].astype(np.uint64)
    
    # Use this insertion idx to determine where to put samples in the idxs_to_pull buffer
    insertion_idxs = np.zeros(len(cell_type_idxs) + 1, dtype = np.uint32)
    insertion_idxs[1:] = np.cumsum(cell_type_counts)
    
    # Create an empty buffer to hold the resulting idxs to pull
    idxs_to_pull = np.zeros(int(cell_type_counts.sum()), dtype = np.int64)
    
    # Iterate across both arrays, idx is what cell based on col, and count is how many to pick
    offset = 0
    
    #for idx, count in zip(cell_type_idxs,cell_type_counts):
    for i in prange(len(cell_type_idxs)):
        
        idx = cell_type_idxs[i]
        count = cell_type_counts[i]
        
        #idxs_to_pull[int(offset):int(offset + count)] = np.random.choice(CELL_TYPE_IDXS_ARR[:,1][CELL_TYPE_IDXS_ARR[:,0] == idx], count, replace = False)
        idxs_to_pull[insertion_idxs[i]:insertion_idxs[i + 1]] = np.random.choice(
            CELL_TYPE_IDXS_ARR[:,1][CELL_TYPE_IDXS_ARR_INDPTR[idx]:CELL_TYPE_IDXS_ARR_INDPTR[idx + 1]], count, replace = False)        
        
    rows = idxs_to_pull
    
    indptr = X_INDPTR
    indices = X_INDICES
    data = X_DATA
    
    M = len(rows)
    new_shape = (M, 28867)

    row_nnz = indptr[rows + 1] - indptr[rows]
    res_indptr = np.zeros(M+1, dtype=np.uint32)
    res_indptr[1:] = np.cumsum(row_nnz)

    nnz = res_indptr[-1]
    res_indices = np.zeros(nnz, dtype = np.uint32)
    res_data = np.zeros(nnz, dtype= np.float32)


    for i in prange(len(rows)):

        row = rows[i]
        start = res_indptr[i]
        stop = res_indptr[i+1]

        res_indices[start:stop] = indices[indptr[row]:indptr[row + 1]]
        res_data[start:stop] = data[indptr[row]:indptr[row + 1]]
        
    # Accumulate
    for i in prange(res_indices.shape[0]):        
        B[res_indices[i]] += res_data[i]
    
    # Compute row means, pass back to reference at B as output.
    B[:] = B / M
    
    

def _threaded_build_and_write_tf_example(args):
    
    # Unroll arguments from zipper
    shardID, (savepath, batch_exp, batch_ct_fracs, batch_total_cells) = args
                
    examples = []
    for exp, ct_frac, total_cells in zip(batch_exp, batch_ct_fracs, batch_total_cells):
            
        tf_feature_exp = uc.tf.utils.training_feature(exp, 
                                                          name = 'expression', dtype = 'float')
        tf_feature_ct_frac = uc.tf.utils.training_feature(ct_frac, 
                                                              name = 'cell_type_fractions', dtype = 'float')
        tf_feature_total_cells = uc.tf.utils.training_feature(np.array([total_cells]), 
                                                                  name = 'total_cells', dtype = 'int')

        tf_example = {}

        tf_example.update(tf_feature_exp)
        tf_example.update(tf_feature_ct_frac)
        tf_example.update(tf_feature_total_cells)

        tf_example = uc.tf.utils.training_example(tf_example)
        examples.append(tf_example)
            
    # Write examples
    uc.tf.utils.write_to_tfrecord(examples, savepath, manual_shardID = shardID, shard_size = 999999999, 
                                      use_manual_shardID = True)     

    
def build_and_write_tf_examples(savepath, exp, ct_fracs, total_cells, shard_batchsize : int = 1500, 
                                   auto_append = True, max_workers = 32):
        """
        Write data shards to GCS bucket via path stem.
        
        :param savepath: path to save file, must bge full path with a target .tfrecords stem.
        :param exp: sample expression data.
        :param ct_fracs: celltype fractions data.
        :param total_cells: total cells in sample data.
        :param shard_batchsize: number of examples to write per data shart, default 1500.
        :param auto_append: whether or not to look for similar stem to savepath in save directory
                                and get the highest shardID and then append on top of it. default true.
        :param max_workers: how many workers to use to write tfrecords. default is 32.                     
        """
        
        # Break up expression, fracs, and total cells into chunks using a generator.
        exp_iter = uc.utils.chunked_np(exp, n = shard_batchsize)
        ct_fracs_iter = uc.utils.chunked_np(ct_fracs, n = shard_batchsize)
        total_cells_iter = uc.utils.chunked_np(total_cells, n = shard_batchsize)
        
        
        if auto_append:
            # Get training shards and sort by shard index
            dfiles = list(filter(lambda x : x.endswith(".tfrecords"), subprocess.check_output(
                "gsutil ls " + os.path.dirname(savepath), 
                shell=True).decode().split("\n")))
            dfiles = sorted(dfiles, key = lambda x : int(x.split("-")[-1].split(".")[0]))
            
            try:
                # Collect the last shard index, add 1, and this is our new shard index offste
                last_idx = int(dfiles[-1].split("-")[-1].split(".")[0])
                next_idx = last_idx + 1
                print(f"[UniCell] Last shard recorded was: {dfiles[-1]}")
                print(f"[UniCell] Last shard index was: {last_idx}, next shard index offset will start at: {next_idx}")
                # Enumerate is the shard ID!
                shard_offset = next_idx
            except:
                print(f"[UniCell] Last shard recorded was: None Found.")
                shard_offset = 0
        else:
            shard_offset = 0
        print(f"[UniCell] Shard offset is: {shard_offset}")
        
        # Create a zipped arguement generator together with the shard offset.
        args = map(lambda x : (x[0] + shard_offset, x[1]), enumerate(zip(repeat(savepath), exp_iter,ct_fracs_iter,total_cells_iter)))
        
        print(f"[UniCell] Beginning writing {int(exp.shape[0] / shard_batchsize)} shards to GCS bucket.")
        # Actually write using all workers to parallelize writing!
        # Use 1 core per batch up to core limit!
        with ProcessPoolExecutor(max_workers = min(max_workers, int(exp.shape[0] / shard_batchsize))) as executor:
            results =  list(progressbar(executor.map(_threaded_build_and_write_tf_example, args), 
                                        max_value = int(exp.shape[0] / shard_batchsize),
                                       suffix = " || Writing TFRecords to GCS Bucket."))
        print(f"[UniCell] Writing shards to GCS bucket complete.")

class PseudoBulkGenerator:
    """
    A PseudoBuilk sample generator.
    
    :param adata: the database object used to build.
    :param obs_key: what column in adata is used as a psuedobulk permutation key.
    :param training_data_size: how many samples to generate.
    :param max_cell_types_per_sample: how many unique cell types in each sample.
    :param max_cells_per_sample: how many cells can be part of a single sample.
    :param max_workers: how many parallel proceses to use.
    :param excluded_codes: which obs_key categorical codes to exclude from sampling.
    """
    
    def __init__(self, 
                 adata : anndata.AnnData,
                 training_data_size : int = 1000000,
                 max_cell_types_per_sample : int = 32,
                 max_cells_per_sample = 10000,
                 max_workers = 128,
                 obs_key : str = 'celltype', 
                 excluded_codes : dict = {0,821,774, 52, 543, 396}):
        
        # Save an internal representation of the data object to prevent CSR object allocations..
        self.X = {
            'data' : adata.X.data, 
            'indices' : adata.X.indices,
            'indptr' : adata.X.indptr,
            'shape' : adata.shape}        
        
        # Break up
        self.X_DATA = self.X['data']
        self.X_INDICES = self.X['indices']
        self.X_INDPTR = self.X['indptr']
        self.X_SHAPE = self.X['shape']
        
        # Remove NaNs from X_DATA
        print(f"[UniCell] Removing NaNs from data array.")
        self.X_DATA[np.isnan(self.X_DATA)] = 0.0
        print(f"[UniCell] Removing NaNs from data array - Successful.")

        # Set constants
        self.TRAINING_DATA_SIZE = training_data_size
        self.MAX_CELL_TYPES_PER_SAMPLE = max_cell_types_per_sample
        self.MAX_CELLS_PER_SAMPLE = max_cells_per_sample
        self.MAX_WORKERS = max_workers
        
        self.excluded_codes = excluded_codes
        
        # Save the actual series information
        self.obs_key = adata.obs[obs_key]
        self.obs_key_values =  adata.obs[obs_key].values
        self.obs_key_categories = adata.obs[obs_key].cat.categories.values
        self.obs_key_codes = adata.obs[obs_key].cat.codes.values
        self.obs_key_value_counts = adata.obs[obs_key].value_counts()
        
        
    def __repr__(self):
        """
        String representation of object.
        """
        
        descr = f"PseudoBulkGenerator object on dataset with: {self.X['shape'][0]:,} cells × {self.X['shape'][1]:,} genes"
        
        return descr
        
    def build_hyperparameters(self):
        """
        Internal utility that builds the hyperparameter space prior to actually pulling the data.
        """
        
        # Temporarily increase the sample size by 1% as we seem to have a ~0.05% rate of zero samples, so over-buffer and replace in end
        self.TRAINING_DATA_SIZE = int(self.TRAINING_DATA_SIZE * 1.01)
        
        
        # Create a numerical representation of the different cell types (0 - 256)
        cell_types = self.obs_key_categories
        

        cell_types_dict = dict(enumerate(cell_types))
        cell_types_dict_inv = {v:k for k,v in cell_types_dict.items()}
        cell_types_as_num = np.arange(0,len(cell_types))
        
        if self.excluded_codes:
            cell_types_as_num_filtered = list(set(np.arange(0,len(cell_types))) - self.excluded_codes)
        else:
            cell_types_as_num_filtered = list(set(np.arange(0,len(cell_types))))
        
        
    
        # Get a numerical representation of the number max cells in each gorup
        cell_counts = self.obs_key_value_counts
        cell_counts.index = cell_counts.index.map(cell_types_dict_inv)
        cell_counts_dict = cell_counts.to_dict()

        cell_types_counts = pd.Series(cell_types_as_num).map(cell_counts).values

        # Create a dict numerical representation of the adata index values for each cell type
        cell_type_idxs = pd.DataFrame([(i, name,np.array(group.index.tolist())) for (i, (name, group)) in \
                                        enumerate(self.obs_key.to_frame().reset_index(drop = True).groupby(self.obs_key.name))], 
                                                  columns = ['num','cell_type','idxs'])
        
        cell_type_idxs['num'] = cell_type_idxs['cell_type'].map(cell_types_dict_inv)
        cell_type_idxs = cell_type_idxs.set_index('num')
        cell_type_idxs_dict = cell_type_idxs.idxs.to_dict()
        
        # Build a numerical representation of each sample and what cells should be in it, and what fraction should be comprised of each cell type
        # The first vector in the tuple is the cell fractions, the second vector in the tuple 
        self.N_UNIQUE_CELL_TYPES_PER_SAMPLE = np.random.randint(1,self.MAX_CELL_TYPES_PER_SAMPLE + 1, size = self.TRAINING_DATA_SIZE)
        self.CELL_COUNT_PER_SAMPLE = np.random.randint(1,self.MAX_CELLS_PER_SAMPLE + 1, size = self.TRAINING_DATA_SIZE)

        
        cell_types_and_fractions = np.zeros((self.TRAINING_DATA_SIZE, cell_types.shape[0]), dtype = np.float16)

        for i in progressbar(range(len(cell_types_and_fractions)), suffix = " || Generating Pseudobulk Sample Templates."):
            # Select which "cells" we will use based on the number for this sample
            # Use a filtered list that does not include unknown or unlabeled cell type.
            selected_cols = np.random.choice(cell_types_as_num_filtered, self.N_UNIQUE_CELL_TYPES_PER_SAMPLE[i], replace = False)

            # Randomly select the fraction distribution for these cells
            selected_col_vals = np.random.randint(1,100, self.N_UNIQUE_CELL_TYPES_PER_SAMPLE[i])

            # Normalize to 1.0
            selected_col_vals = np.round(selected_col_vals / selected_col_vals.sum(), 3)

            # Get the actual count of cells and round to the nearest whole number
            selected_col_vals *= self.CELL_COUNT_PER_SAMPLE[i]
            selected_col_vals = np.floor(selected_col_vals)

            # Assign values to the row in the data buffer 
            cell_types_and_fractions[i, selected_cols] = selected_col_vals

        # Ensure that no cell sampling contains more than the number of cells in our database
        cell_types_and_fractions = np.minimum(cell_types_and_fractions, cell_types_counts)
        
        
        # Set the training data size back
        self.TRAINING_DATA_SIZE = int(self.TRAINING_DATA_SIZE / 1.01)
        
        # Clip the cell_types and fractions such that any zero counts are removed and we clip down to our required size
        cell_types_and_fractions = cell_types_and_fractions[cell_types_and_fractions.sum(1) != 0][0:self.TRAINING_DATA_SIZE]
        

        # Unpack our cell type idxs dict in such a way that make for fast selection
        # and allow us to pass it into JIT
        CELL_TYPE_IDXS_ARR = np.concatenate([np.vstack([np.ones(v.shape[0])*k, v]).T.astype(np.uint64) for k,v in cell_type_idxs_dict.items()])
        CELL_TYPE_IDXS_ARR_INDPTR = []

        # Use a CSR-like indptr to enable fast access of all cell type index addresses
        for i in progressbar(range(self.obs_key_categories.shape[0]), suffix = " || Generating Cell Type Index Pointers."):
            amin = np.argwhere(CELL_TYPE_IDXS_ARR[:,0] == i).min()
            amax = np.argwhere(CELL_TYPE_IDXS_ARR[:,0] == i).max() + 1
            if i == 0: CELL_TYPE_IDXS_ARR_INDPTR.append(amin)
            CELL_TYPE_IDXS_ARR_INDPTR.append(amax)

        CELL_TYPE_IDXS_ARR_INDPTR = np.array(CELL_TYPE_IDXS_ARR_INDPTR, dtype = np.uint64)
        
        # Assign final results to class variables
        self.CELL_TYPES_AND_FRACTIONS = cell_types_and_fractions
        self.CELL_TYPE_IDXS_ARR = CELL_TYPE_IDXS_ARR
        self.CELL_TYPE_IDXS_ARR_INDPTR = CELL_TYPE_IDXS_ARR_INDPTR
        
        
    def reduce_sample(self, sample_row, B):
        """
        Actually generates a sample using the cell_types_and_Fractions row information index.
        """
        
        return _get_sample_as_mean_exp(B, sample_row, 
                            self.CELL_TYPE_IDXS_ARR,
                            self.CELL_TYPE_IDXS_ARR_INDPTR,
                            self.X_INDPTR, self.X_INDICES, self.X_DATA)
    
    def _threaded_reduce_sample(self, args):
        """
        Internal helper function to be called 
        """
        return self.reduce_sample(args[0], args[1])
        
    
    def reduce_samples(self, sample_rows = np.empty(0), B = np.empty(0), n_threads : int = 24, half_precision = False):
        """
        Utility that accepts an array of sample rows (cell_types_and_fractions) and will
        return a buffer containing processed samples.
        
        :param sample_rows: cell_types_and_fractions, if none passed, defaults to internal cell_types_and_fracs, otherwise use external
        :param B: result buffer,, an ndarray, if not passed, automatically generated based on the sample_rows shape.
        :param n_threads: number of threads to assign to threadpool for calling each inidivudal sample reduction. default 32.
        :param half_precision: save results as float16 to save memory.
        
        :return B: the sample buffer of results.
        """
                       
        
        
        # If we don't passed a specific sample_rows arguement, use internal.
        if sample_rows.size == 0:
            sample_rows = self.CELL_TYPES_AND_FRACTIONS
                       
        # Determine output dtype based on half-precision flag.
        B_dtype = np.float16 if half_precision else np.float32
        
        # If empty array passed make one
        # Have to use float32 for numba support.
        if B.size == 0:
            B = np.zeros((sample_rows.shape[0], 28867), dtype = np.float32)
        
        # Iterate sample_rows to reduce to samples
        # Use a ThreadPool to queue up function calls to avoid overhead here
        
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers = n_threads) as executor:
                _ = list(progressbar(executor.map(self._threaded_reduce_sample, zip(sample_rows, B)), max_value = sample_rows.shape[0]))
        else:
            for args in progressbar(zip(sample_rows, B), max_value = sample_rows.shape[0]):
                _ = self._threaded_reduce_sample(args)
        
        # Return the buffer, or save it. Also return normalized sample rows and total cell counts.
        total_cells = sample_rows.sum(1)
        
        return B.astype(B_dtype), (sample_rows / total_cells.reshape(-1,1)).astype(B_dtype), total_cells.astype(np.int32)