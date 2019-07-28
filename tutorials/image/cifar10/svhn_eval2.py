from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from six.moves import urllib

import numpy as np
import tensorflow as tf

import cifar10
import os
import sys
import tarfile
import svhn_readInput

from pathlib import Path, PureWindowsPath


#rename file to eval??
#Qualcosa non va!! immagini sbagliate ecc...

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/svhn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/svhn_train',
                           """Directory where to read model checkpoints.""")

#eval interval:
#tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")

#Eval examples
#tf.app.flags.DEFINE_integer('num_examples', 10000,
#                            """Number of examples to run.""")

tf.app.flags.DEFINE_integer('num_examples', 6000,
                            """Number of examples to run.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
#tf.app.flags.DEFINE_integer('batch_size', 64,
 #                           """Number of images to process in a batch.""")

batch_size = 128

data_dir = '/tmp/svhn_data'
data_dirDigits = '/tmp/svhn_dataDigitsEval'
eval_dir = '/tmp/svhn_eval'
DATA_URL = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: step: %s, precision @ 1 = %.3f' % (datetime.now(), global_step, precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    #images, labels = cifar10.inputs(eval_data=eval_data) -> altrimenti legge i cifar 10..
    dir = Path(data_dirDigits)

    images, labels = svhn_readInput.elaborateInput(True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


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
  extracted_dir_path = os.path.join(data_dir, 'test')
  if not os.path.exists(extracted_dir_path):
    print("extracting images.. it takes some time")
    tarfile.open(filepath, 'r:gz').extractall(data_dir)


def main(argv=None):  # pylint: disable=unused-argument
  maybe_download_and_extract()
  if tf.gfile.Exists(eval_dir):
    tf.gfile.DeleteRecursively(eval_dir)
  tf.gfile.MakeDirs(eval_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()