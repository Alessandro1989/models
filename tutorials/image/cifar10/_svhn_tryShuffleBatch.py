import tensorflow as tf
import numpy as np

def main():
    data = np.arange(1, 100 + 1)
    data_input = tf.constant(data)

    batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=10, capacity=100,
                                           min_after_dequeue=10, allow_smaller_final_batch=True)
    batch_no_shuffle = tf.train.batch([data_input], enqueue_many=True, batch_size=10, capacity=100,
                                      allow_smaller_final_batch=True)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print(i, sess.run([batch_shuffle, batch_no_shuffle]))
        coord.request_stop()
        coord.join(threads)

def main2():
    data = np.arange(1, 100 + 1)
    data_input = tf.constant(data)

    #batch_shuffle = tf.train.shuffle_batch([data_input], enqueue_many=True, batch_size=10, capacity=100,
      #                                     min_after_dequeue=10, allow_smaller_final_batch=True)

    data_input = tf.constant(np.arange(2, 100 + 1))
    batch_shuffle= tf.train.shuffle_batch([data_input], batch_size=10, num_threads = 16, capacity = 3000 + 30, min_after_dequeue = 50)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            print(i, sess.run([batch_shuffle]))
        coord.request_stop()
        coord.join(threads)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

if __name__ == '__main__':
  #main()
  main2()
