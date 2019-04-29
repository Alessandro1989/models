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
from tensorflow.python.ops import array_ops
import svhn_readInputTrain

data_dir = '/tmp/svhn_data'
train_dir = '/tmp/svhn_train'
data_dirDigits = '/tmp/svhn_dataDigits'
DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
batch_size = 128 #number of images to process in a batch
IMAGE_SIZE = 24

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



  #crop digits if is necessary
  dir = Path(data_dirDigits)
  if (not dir.exists()):
    svhn_readInputTrain.readInfoAndCropDigits()

  #pathDataDir = Path(data_dir, 'train')
  pathDataDir = Path(data_dirDigits)
  filenames = list(pathDataDir.glob('*.png'))
  # converte in a list of string paths
  filenames = list(map(lambda x: str(x.absolute()), filenames))


  # Create a queue that produces the filenames to read
  # (he converts the strings in tensors) and add them to the fifoqueue
  filename_queue = tf.train.string_input_producer(filenames)
  reader = tf.WholeFileReader("reader")
  #restituisce una stringa che rappresenta il contenuto, e una stringa per il filename
  key,value = reader.read(filename_queue, "read")
  img_u = tf.image.decode_jpeg(value, channels=3)
  img_f = tf.cast(img_u, tf.float32)
  #img_4 = tf.expand_dims(img_f, 0)

  #4-D Tensor of shape [batch, height, width, channels] ?? channels = 3 , e altezza e larghezze delle immagini???
  #img_f = tf.image.random_flip_left_right(img_f) # must not used NA
  img_f = tf.image.random_brightness(img_f, max_delta=60000)  #63
  img_f = tf.image.random_contrast(img_f, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(img_f)
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  # Set the shapes of tensors.
  #float_image.set_shape([height, width, 3]) #doesnt work!!-> seems just for infos.. it's not our case

  float_image = tf.image.resize_image_with_pad(float_image, height, width)
  img_4 = tf.expand_dims(float_image, 0)


  #height = IMAGE_SIZE
  #width = IMAGE_SIZE
  # Set the shapes of tensors.
  #float_image.set_shape([height, width, 3])

  #Questo forse è meglio:
  #[{'label': '3', 'top': '7', 'left': '52', 'width': '21', 'height': '46'}, {'label': '1', 'top': '10', 'left': '74', 'width': '15', 'height': '46'}]
  #ffsetHeight = tf.placeholder(tf.int32)
  #offsetWith = tf.placeholder(tf.int32)
 # targetHeight = tf.placeholder(tf.int32)
  #targetWidth = tf.placeholder(tf.int32)

  #Forse ci serve per il label
  #pngname = key.eval().decode("utf-8").split("\\")[-1] -> operation?
  #offsetWith = [digitsInfo[key][0]['left']] -> operation?
  #devi cambiare questi in operazioni..
  #stringkey = tf.cast(key,tf.string)
  #splitPng = tf.strings.split(stringkey, "\\")
  #pngName = tf.slice(splitPng, splitPng.getShape()[0]-1, 1)
  #stringkey = tf.compat.as_text(key)
  #stringkey = tf.convert_to_tensor(key, dtype=tf.string)

  #keyexpand = tf.expand_dims(key,0)
  #splitPng = tf.strings.split(keyexpand, "\\")
  #png = splitPng[-1]
  #to fix : 4 -> with dimshape()-1
  #png = tf.sparse_slice(splitPng,[0,4],[1,1])

  img_opsummary = tf.summary.image("img", img_4)

  #sess = tf.InteractiveSession()
  #tf.global_variables_initializer().run()
  #imgop_sess = sess.run(img_opsummary)
  #train_writer = tf.summary.FileWriter(train_dir, sess.graph)
  #train_writer.add_summary(imgop_sess)
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
   # print(sess.run(key))
   # print(sess.run(splitPng))
   # print(sess.run(png))


    #print(sess.run(key))
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    #for i in range(1,len(filenames)):
    for i in range(1, 20):
      #perche solo 10 immagini per "slot"? (facendo così va un po' meglio ma mica tatno però!)
      print(i)
      #print(key.eval())
      img_opsummary = tf.summary.image(str(i), img_4, 1000)
      for i in range(1,15):

        #pngname = key.eval().decode("utf-8").split("\\")[-1]
        imgop_sess = sess.run(img_opsummary) #need label together..
        train_writer.add_summary(imgop_sess)

    coord.request_stop()
    coord.join(threads)

  #https: // stackoverflow.com / questions / 34696845 / how - to - see - multiple - images - through - tf - image - summary




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
