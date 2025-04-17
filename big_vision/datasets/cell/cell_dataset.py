import tensorflow as tf
import big_vision.datasets.core as ds_core
import jax
import functools
import tensorflow as tf
import overrides
import tqdm

from big_vision.datasets.cell.cell_config import N_TIMESTAMPS, N_FEATURES, DATA_DIR, LOOKUP_DIR
import big_vision.datasets.cell.cell_dataset_utils as cdu

class CellDataSource(ds_core.DataSource):
    def __init__(self, split, data_dir=DATA_DIR, lookup_dir=LOOKUP_DIR):

        """
        Args:
            name: Name of the dataset.
            split: Split of the dataset.
            data_dir: Directory of the dataset.
            lookup_dir: Directory of the lookup dataset.
            In our use case, data_dir contains input_ids and lookup_dir maps the input_ids to the cell data.
        """
        if data_dir is None:
            raise ValueError("data_dir must be provided")
        
        if lookup_dir is None:
            raise ValueError("lookup_dir must be provided")

        self.data_dir = data_dir
        self.lookup_dir = lookup_dir
        
        train_dataset, eval_dataset = cdu.load_cell_datasets(data_dir)

        print(f"Train dataset shape: {train_dataset.shape}")
        print(f"Eval dataset shape: {eval_dataset.shape}")

        if split == "train":
            self.dataset = train_dataset
        elif split == "validation" or split == "eval":
            self.dataset = eval_dataset
        else:
            raise ValueError(f"Invalid split: {split}")

        print(f"Loading lookup dataset from: {lookup_dir}")
        self.lookup_dataset = cdu.load_lookup_dataset(lookup_dir)

        print(f"Lookup dataset shape: {self.lookup_dataset.shape}")

        print(f"Loading generator function")
        self.generator_fn = cdu.cell_data_generator
 
        print(f"Generating output signature")
        self.output_signature = cdu.generate_output_signature()

       #  self.total_examples = self._count_examples()
        self.total_examples = 180000

        print(f"Setting up process splits")
        self._setup_process_splits()

        print(f"Num examples per process: {self.num_examples_per_process}")
        print(f"Process range: {self.process_range}")
        print(f"Initialized CellDataSource")

    @property
    def total_examples(self):
        """Total number of examples across all processes."""
        return self._total_examples

    @total_examples.setter
    def total_examples(self, value):
        self._total_examples = value

    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, value):
        self._dataset = value
    
    def num_examples_per_process(self):
        """List of number of examples per process."""
        return self._examples_per_process
    
    @overrides.overrides
    def get_tfdata(self, ordered=False, *, process_split=True, allow_cache=True, **kw):
        return (self._cached_get_dataset if allow_cache else self._get_dataset)(ordered=ordered, process_split=process_split, **kw)
    
    def _setup_process_splits(self):
        """Calculate process-specific data splits."""
        n_processes = jax.process_count()
        current_process = jax.process_index()

        # The first remainder processes get one more example
        base_count = self._total_examples // n_processes
        remainder = self._total_examples % n_processes
        
        start_idx = current_process * base_count
        if current_process < remainder:
            start_idx += current_process
            base_count += 1
        else:
            start_idx += remainder
            
        self.process_range = (start_idx, start_idx + base_count)
        
        # Store num_examples_per_process
        self._examples_per_process = [
            base_count + (1 if i < remainder else 0)
            for i in range(n_processes)
        ]

    @functools.cache
    def _count_examples(self):
        """Count total examples in the generator dataset."""
        count = 0
        for _ in tqdm.tqdm(self.generator_fn(self.dataset, self.lookup_dataset), 
                         desc="Counting examples",
                         total=self.dataset.shape[0]):
            count += 1
        return count

    def _get_dataset(self, ordered=False, *, process_split=True, **kw):
        """Get the dataset for this process.
        
        Args:
            ordered: Whether to maintain deterministic ordering
            process_split: Whether to use process-specific split
            allow_cache: Whether to cache the dataset
        """

        # generator for the current process
        def _process_generator():
            start_idx, end_idx = self.process_range
            for idx, example in enumerate(self.generator_fn(self.dataset, self.lookup_dataset)):
                if start_idx <= idx < end_idx:
                    yield example

        # Wrapper for the generator function that provides the required arguments
        def _generator_wrapper():
            for example in self.generator_fn(self.dataset, self.lookup_dataset):
                yield example

        # Create dataset with proper signature
        if process_split:
            ds = tf.data.Dataset.from_generator(
                _process_generator,
                output_signature=self.output_signature
            )
        else:
            ds = tf.data.Dataset.from_generator(
                _generator_wrapper,
                output_signature=self.output_signature
            )

        # Apply shuffling if not ordered
        if not ordered:
            ds = ds.shuffle(buffer_size=min(1000, self._total_examples))
        
        ds = ds.repeat()
        return ds

    _cached_get_dataset = functools.cache(_get_dataset)

if __name__ == "__main__":
    # Create a dataset
    data_source = CellDataSource(split="train", data_dir=DATA_DIR, lookup_dir=LOOKUP_DIR)
    print("Number of examples per process:", data_source.num_examples_per_process)
    print("Total examples:", data_source.total_examples)
    
    dataset = data_source.get_tfdata(ordered=False, process_split=False, allow_cache=False)
    
    # Take a look at some examples
    print("\nFirst 10 examples:")
    for example in dataset.take(20):
        print("Example shape:", example["image"][0][:20])
        print("Example dtype:", example["image"].dtype)
        print("---")
    
    # Look at batched examples
    print("\nBatched examples:")
    for batch in dataset.take(10).batch(10):
        print("Batch shape:", batch["image"].shape)
        print("Batch dtype:", batch["image"].dtype)
        print("---")
