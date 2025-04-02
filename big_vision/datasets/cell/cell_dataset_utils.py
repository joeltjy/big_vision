from datasets import load_from_disk
import anndata as ad
import numpy as np
import tensorflow as tf
from cell_config import N_TIMESTAMPS, N_FEATURES
import tqdm

def load_cell_datasets(data_dir):
    """
    Load cell dataset from the given directory. 
    Each data point is a 3-tuple where the first element is a list of IDs.
    The actual data is stored in another dataset where the IDs are the keys.
    """
    try:
        print("Loading dataset...")
        dataset = tqdm.tqdm(load_from_disk(data_dir))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {data_dir} not found")
    

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    print(f"Loaded {train_dataset.shape[0]} train and {eval_dataset.shape[0]} eval examples")

    return train_dataset, eval_dataset

def load_lookup_dataset(lookup_dir):
    """
    Load lookup dataset from the given directory.
    Each ID points to a 19089-length vector.
    """
    try:
        print("Loading lookup dataset...")
        adata = tqdm.tqdm(ad.read_h5ad(lookup_dir, backed='r'))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {lookup_dir} not found")

    print(f"Loaded dataset of size {adata.shape}")

    return adata

def cell_data_generator(dataset, lookup_dataset):
    """
    Generator to load cell data. 
    The dataset is a 3-tuple where the first element is a list of IDs.
    The actual data is stored in another dataset where the IDs are the keys.
    """
    n_dataset_entries = dataset.shape[0] # 180000

    for idx in range(n_dataset_entries):
        dataset_ids = dataset[idx] 
        assert len(dataset_ids) == N_TIMESTAMPS
        dataset_data = np.array([lookup_dataset[id] for id in dataset_ids]) # (N_TIMESTAMPS, N_FEATURES)
        yield {"cell_data": dataset_data}

def generate_output_signature():
    output_signature = {
        "cell_data": tf.TensorSpec(shape=(N_FEATURES, N_TIMESTAMPS), dtype=tf.float32),
    }
    return output_signature

    