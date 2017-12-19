import tensorflow as tf

# Global parameters
DATA_FILE = 'boston_housing.csv'
BATCH_SIZE = 10
NUM_FEATURES = 14

def data_generator(filename):
    """
    Generates Tensors in batches of size Batch_SIZE.
	Args: String Tensor
	Filename from which data is to be read
	Returns: Tensors
	feature_batch and label_batch
    """

    f_queue = tf.train.string_input_producer(filename)
    reader = tf.TextLineReader(skip_header_lines=1) # Skips the first line
    _, value = reader.read(f_queue)

    # next we declare the default value to use in case a data is missing
    record_defaults = [ [0.0] for _ in range(NUM_FEATURES)]
    data = tf.decode_csv(value, record_defaults=record_defaults)
    condition = tf.equal(data[13], tf.constant(50.0))
    data = tf.where(condition, tf.zeros(NUM_FEATURES), data[:])
    features = tf.stack(tf.gather_nd(data,[[5],[10],[12]]))
    label = data[-1]

    # minimum number elements in the queue after a dequeue
    min_after_dequeue = 10 * BATCH_SIZE
    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    feature_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE,
                                                     capacity=capacity, min_after_dequeue=min_after_dequeue)

    return feature_batch, label_batch

def generate_data(feature_batch, label_batch):
    with tf.Session() as sess:
        # intialize the queue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(5): # Generate 5 batches
            features, labels = sess.run([feature_batch, label_batch])
            print (features, "HI")
        coord.request_stop()
        coord.join(threads)


if __name__ =='__main__':
    feature_batch, label_batch = data_generator([DATA_FILE])
    generate_data(feature_batch, label_batch)