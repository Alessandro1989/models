import tensorflow as tf
import os
import sys
from six.moves import urllib
import tarfile
import datetime

import svhn_readInput
from datetime import datetime
import time

import tensorflow as tf

from enumTypeSet import  TypeSet

import svhn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

data_dir = '/tmp/svhn_data'
train_dir = '/tmp/svhn_train'

DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
IMAGE_SIZE = svhn_readInput.IMAGE_SIZE

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = svhn_readInput.NUM_CLASSES
BATCH_SIZE = svhn_readInput.BATCH_SIZE

def main(argv=None):
  maybe_download_and_extract()
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  train()


def train():
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step() #"(un contatore per il training.."
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      #images, labels = elaborateInput()
      images, labels = svhn_readInput.elaborateInput(TypeSet.TRAIN)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = svhn.inference(images)

    #Calculate loss.
    loss = svhn.loss(logits, labels) #Problems with cast con labels in int64

    #Build a Graph that trains the model with one batch of examples and
    #updates the model parameters.
    train_op = svhn.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * BATCH_SIZE / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement),
         save_checkpoint_secs=60) as mon_sess: #save every 60 seconds
      while not mon_sess.should_stop():
        #mon_sess.run(train_op) (like sess.run, ci interessano le immagini che sta elaborando per capire dove sbaglia
      #  try:
          mon_sess.run(train_op)
      #  except: #resize exception.. (there is some images give problems..)
      #    print('error')
       #   input("write something to continue: ")


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
