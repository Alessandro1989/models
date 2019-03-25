import tensorflow as tf
import os
import sys
from six.moves import urllib
import tarfile
import datetime
from pathlib import Path, PureWindowsPath
import cifar10
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner

data_dir = '/tmp/svhn_data'
train_dir = '/tmp/svhn_train'
DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
batch_size = 128 #number of images to process in a batch

def main(argv=None):
  maybe_download_and_extract()
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  train()


def train():
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step() #"(un contatore per il training.."

  #ci concentriamo sull'input e basta per adesso..
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #images, labels = elaborateInput()
      elaborateInput()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = cifar10.inference(images)

    # Calculate loss.
    #loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = cifar10.train(loss, global_step)

    # class _LoggerHook(tf.train.SessionRunHook):
    #   """Logs loss and runtime."""
    #
    #   def begin(self):
    #     self._step = -1
    #     self._start_time = time.time()
    #
    #   def before_run(self, run_context):
    #     self._step += 1
    #     return tf.train.SessionRunArgs(loss)  # Asks for loss value.
    #
    #   def after_run(self, run_context, run_values):
    #     if self._step % FLAGS.log_frequency == 0:
    #       current_time = time.time()
    #       duration = current_time - self._start_time
    #       self._start_time = current_time
    #
    #       loss_value = run_values.results
    #       examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
    #       sec_per_batch = float(duration / FLAGS.log_frequency)
    #
    #       format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
    #                     'sec/batch)')
    #       print (format_str % (datetime.now(), self._step, loss_value,
    #                            examples_per_sec, sec_per_batch))
    #
    # with tf.train.MonitoredTrainingSession(
    #     checkpoint_dir=FLAGS.train_dir,
    #     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
    #            tf.train.NanTensorHook(loss),
    #            _LoggerHook()],
    #     config=tf.ConfigProto(
    #         log_device_placement=FLAGS.log_device_placement)) as mon_sess:
    #   while not mon_sess.should_stop():
    #     mon_sess.run(train_op)


def elaborateInput():
  """Construct distorted input for SVHN training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  """

  pathDataDir = Path(data_dir, 'train')
  filenames = list(pathDataDir.glob('*.png'))
  #converte in a list of string paths
  filenames = list(map(lambda x: str(x.absolute()), filenames))
  # Create a queue that produces the filenames to read
  # (he converts the strings in tensors) and add them to the fifoqueue
  filename_queue = tf.train.string_input_producer(filenames)
  reader = tf.WholeFileReader("reader")
  #restituisce una stringa che rappresenta il contenuto, e una stringa per il filename
  key,value = reader.read(filename_queue, "read")
  img_u = tf.image.decode_jpeg(value, channels=3)
  img_f = tf.cast(img_u, tf.float32)
  img_4 = tf.expand_dims(img_f, 0)
  #img_i = tf.decode_raw(value, tf.int8) #w

  #img_f = tf.cast(img_i, tf.float32)
  #img_f_reshape = tf.reshape();
  #img_4 = tf.expand_dims(img_f,0)





  #another stuff:
  #strpathImg3 = str(filenames[2].absolute())
  #stringPath_tensor = ops.convert_to_tensor(strpathImg3, dtype=dtypes.string)
  #img_b = tf.read_file(stringPath_tensor)
  #img_u = tf.image.decode_jpeg(img_b, channels=3)
 # img_f = tf.cast(img_u, tf.float32)
  #img_4 = tf.expand_dims(img_f,0)
  #channels = 3

  img_opsummary = tf.summary.image("img", img_4)

  #sess = tf.InteractiveSession()
  #tf.global_variables_initializer().run()
  #imgop_sess = sess.run(img_opsummary)
  #train_writer = tf.summary.FileWriter(train_dir, sess.graph)
  #train_writer.add_summary(imgop_sess)
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run(key))
    print(sess.run(value))
    print(sess.run(img_f))
    #attenzione: facendo così, lo shape di prova è 1D.. come fai a farlo 4d? e l'immagine puoi rappresentarla 1d? cosa rappresenta un numero? bohh
    imgop_sess = sess.run(img_opsummary)
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    train_writer.add_summary(imgop_sess)
    #for i in range(1,len(filenames)):
    for i in range(1, 20):
      #perche solo 10 immagini per "slot"? (facendo così va un po' meglio ma mica tatno però!)
      print(i)
      img_opsummary = tf.summary.image(str(i), img_4, 1000)
      for i in range(1,15):
        imgop_sess = sess.run(img_opsummary)
        train_writer.add_summary(imgop_sess)


    coord.request_stop()
    coord.join(threads)

  #https: // stackoverflow.com / questions / 34696845 / how - to - see - multiple - images - through - tf - image - summary


  #images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
  #                                                batch_size=FLAGS.batch_size)

  # constucting distorted input for training using the reader ops (ATTENZION abbiamo immagini non file binari, quindi?)

  #   filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
  #                for i in xrange(1, 6)]
  #   for f in filenames:
  #     if not tf.gfile.Exists(f):
  #       raise ValueError('Failed to find file: ' + f)
  #
  #   # Create a queue that produces the filenames to read.
  #   filename_queue = tf.train.string_input_producer(filenames)
  #
  #   with tf.name_scope('data_augmentation'):
  #     # Read examples from files in the filename queue.
  #     read_input = read_cifar10(filename_queue)
  #     reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  #
  #     height = IMAGE_SIZE
  #     width = IMAGE_SIZE
  #
  #     # Image processing for training the network. Note the many random
  #     # distortions applied to the image.
  #
  #     # Randomly crop a [height, width] section of the image.
  #     distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  #
  #     # Randomly flip the image horizontally.
  #     distorted_image = tf.image.random_flip_left_right(distorted_image)
  #
  #     # Because these operations are not commutative, consider randomizing
  #     # the order their operation.
  #     # NOTE: since per_image_standardization zeros the mean and makes
  #     # the stddev unit, this likely has no effect see tensorflow#1458.
  #     distorted_image = tf.image.random_brightness(distorted_image,
  #                                                  max_delta=63)
  #     distorted_image = tf.image.random_contrast(distorted_image,
  #                                                lower=0.2, upper=1.8)
  #
  #     # Subtract off the mean and divide by the variance of the pixels.
  #     float_image = tf.image.per_image_standardization(distorted_image)
  #
  #     # Set the shapes of tensors.
  #     float_image.set_shape([height, width, 3])
  #     read_input.label.set_shape([1])
  #
  #     # Ensure that the random shuffling has good mixing properties.
  #     min_fraction_of_examples_in_queue = 0.4
  #     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
  #                              min_fraction_of_examples_in_queue)
  #     print('Filling queue with %d CIFAR images before starting to train. '
  #           'This will take a few minutes.' % min_queue_examples)
  #
  #   # Generate a batch of images and labels by building up a queue of examples.
  #   return _generate_image_and_label_batch(float_image, read_input.label,
  #                                          min_queue_examples, batch_size,
  #                                          shuffle=True)
  # return images, labels


def maybe_download_and_extract():
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(data_dir, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                       float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(data_dir, 'train')
  if not os.path.exists(extracted_dir_path):
    print("extracting images.. it takes some time")
    tarfile.open(filepath, 'r:gz').extractall(data_dir)


if __name__ == '__main__':
  tf.app.run() #main()
