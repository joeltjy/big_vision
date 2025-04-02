import os

# Dataset dimensions
N_TIMESTAMPS = 39
N_FEATURES = 19089

def get_project_root():
    """Get the project root directory by going up 3 levels from this file."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def find_dataset_dir():
    """Find the dataset directory using environment variables or default paths."""
    # Try environment variable first
    data_dir = os.getenv('CELL_DATASET_DIR')
    if data_dir:
        return data_dir

    project_root = get_project_root()
    
    # If in snap container, try common workspace locations
    if 'snapd-desktop-integration' in project_root:
        for path in [
            os.path.join(os.path.expanduser('~'), 'main_page', 'big_vision', 'cell_dataset'),
            os.path.join(os.path.expanduser('~'), 'big_vision', 'cell_dataset'),
            os.path.join(os.path.expanduser('~'), 'cell_dataset'),
        ]:
            if os.path.exists(path):
                return path

    # Default to project root
    return os.path.join(project_root, "cell_dataset")

def get_lookup_dir():
    """Get the lookup file path using environment variables or default path."""
    lookup_dir = os.getenv('CELL_LOOKUP_DIR')
    if lookup_dir:
        return lookup_dir
    return os.path.join(DATA_DIR, "reprogramming_schiebinger_serum_computed.h5ad")

# Initialize paths
DATA_DIR = find_dataset_dir()
LOOKUP_DIR = get_lookup_dir()

# Print configuration for debugging
if __name__ == "__main__":
    print(f"Dataset dimensions: {N_TIMESTAMPS} timestamps, {N_FEATURES} features")
    print(f"Data directory: {os.path.abspath(DATA_DIR)}")
    print(f"Lookup file: {os.path.abspath(LOOKUP_DIR)}")

