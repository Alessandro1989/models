from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import svhn

import svhn_readInput


FLAGS = tf.app.flags.FLAGS


#eval interval:
tf.app.flags.DEFINE_integer('eval_interval_secs', 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
NUM_EXAMPLE = svhn_readInput.NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION




BATCH_SIZE = svhn_readInput.BATCH_SIZE

validation_dir = '/tmp/svhn_validation'
data_dir = svhn_readInput.data_dir
data_dirDigits = svhn_readInput.data_dirDigitsTrain
checkpoint_dir = '/tmp/svhn_train'




def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
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

      num_iter = int(math.ceil(NUM_EXAMPLE / BATCH_SIZE))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * BATCH_SIZE
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
    images, labels = svhn_readInput.elaborateFilesValidation()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = svhn.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        svhn.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(validation_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


"""
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
"""

def main(argv=None):
  #maybe_download_and_extract() -> it use the data from the training set
  if tf.gfile.Exists(validation_dir):
    tf.gfile.DeleteRecursively(validation_dir)
  tf.gfile.MakeDirs(validation_dir)
  evaluate()

if __name__ == '__main__':
  tf.app.run()
