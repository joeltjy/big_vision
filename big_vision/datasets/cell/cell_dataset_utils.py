from datasets import load_from_disk
import anndata as ad
import numpy as np
import tensorflow as tf
import os
from big_vision.datasets.cell.cell_config import N_TIMESTAMPS, N_FEATURES

def load_cell_datasets(data_dir):
    """
    Load cell dataset from the given directory. 
    Each data point is a 3-tuple where the first element is a list of IDs.
    The actual data is stored in another dataset where the IDs are the keys.
    """
    # Convert to absolute path and verify it exists
    data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    try:
        # Load train and test datasets separately
        train_dataset = load_from_disk(os.path.join(data_dir, 'train'))
        test_dataset = load_from_disk(os.path.join(data_dir, 'test'))
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {data_dir}: {str(e)}")

    return train_dataset, test_dataset

def load_lookup_dataset(lookup_dir):
    """
    Load lookup dataset from the given directory.
    Each ID points to a 19089-length vector.
    """
    try:
        # Convert to absolute path and verify it exists
        lookup_dir = os.path.abspath(lookup_dir)
        if not os.path.exists(lookup_dir):
            raise FileNotFoundError(f"Lookup file not found: {lookup_dir}")
            
        adata = ad.read_h5ad(lookup_dir, backed='r')
        # Convert to numpy array to avoid repeated indexing issues
        lookup_data = adata.X[:]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset {lookup_dir} not found")
    except Exception as e:
        raise RuntimeError(f"Failed to load lookup dataset from {lookup_dir}: {str(e)}")

    return lookup_data

def cell_data_generator(dataset, lookup_dataset):
    """
    Generator to load cell data. 
    The dataset is a 3-tuple where the first element is a list of IDs.
    The actual data is stored in another dataset where the IDs are the keys.
    """
    n_dataset_entries = dataset.shape[0] # 180000

    for idx in range(n_dataset_entries):
        dataset_ids = dataset[idx]["input_ids"] 
        assert len(dataset_ids) == N_TIMESTAMPS
        # Convert sparse matrices to dense arrays
        dataset_data = np.array([lookup_dataset[id].toarray().flatten() for id in dataset_ids]).reshape(N_TIMESTAMPS, N_FEATURES, 1) # (N_TIMESTAMPS, N_FEATURES, 1)
        if idx == 0:
            print("dataset_data shape", dataset_data.shape)
        for i in range(N_TIMESTAMPS):
            yield {"image": dataset_data[i:i+1]}

def generate_output_signature():
    output_signature = {
        "image": tf.TensorSpec(shape=(1, N_FEATURES, 1), dtype=tf.float32),
    }
    return output_signature

    