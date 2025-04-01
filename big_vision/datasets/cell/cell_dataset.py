import tensorflow as tf

################################################################################
#                                                                              #
#                               Helper functions                               #
#                                                                              #
################################################################################

# generator to load cell data
def my_generator():
    for i in range(100):
        yield {"image": tf.random.uniform((224, 224, 3)), "label": i % 10}

# Define output structure
output_signature = {
    "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
}

# Create a dataset
dataset = tf.data.Dataset.from_generator(my_generator, output_signature=output_signature)