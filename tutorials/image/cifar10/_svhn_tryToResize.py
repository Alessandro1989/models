from __future__ import print_function
from PIL import Image
import os
import sys
from pathlib import Path, PureWindowsPath
import tensorflow as tf
import os
import sys
from six.moves import urllib
import tarfile
import datetime
from pathlib import Path, PureWindowsPath
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


import tensorflow as tf


data_dir = '/tmp/svhn_data'
train_dir = '/tmp/svhn_train'
data_dirDigitsTrain = '/tmp/svhn_dataDigits'
data_dirDigitsEval = '/tmp/svhn_dataDigitsEval'

IMAGE_SIZE = 24



def main():
    tryToResize2()
    #tryToResize2()

    print("main readinput")
    #readInfoAndCropDigits()
    #da cambiare e refactoring
    #readInfoAndCropDigitsEval()
    #read_input_train()

def tryToResize2(eval=False):
  dir = Path(data_dirDigitsTrain)

  sess = tf.InteractiveSession()
  #pathDataDir = Path(data_dir, 'train')
  #pathDataDir = Path(data_dirDigitsTrain)

  filenames = list(dir.glob('*.png'))
  # converte in a list of string paths
  filenames = list(map(lambda x: str(x.absolute()), filenames))
  #filenames = ["C:\\tmp\\svhn_dataDigits\\12189_1.png" for x in range(67000)]

  
  # Create a queue that produces the filenames to read
  # (he converts the strings in tensors) and add them to the fifoqueue
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
  reader = tf.WholeFileReader("reader")
  key,value = reader.read(filename_queue, "read")
  img_u = tf.image.decode_jpeg(value, channels=3)
  img_f = tf.cast(img_u, tf.float32)
  #img_f = tf.image.random_brightness(img_f, max_delta=60000)  #63
  #img_f = tf.image.random_contrast(img_f, lower=0.2, upper=1.8)
  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(img_f)
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  float_image = tf.image.resize_image_with_pad(img_f, height, width)


  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(len(filenames)):
      try:
        result = sess.run([key, float_image])
        print(i,result)
      except:
        print('error in: ', filenames[i])
        input("write something to continue: ")

    coord.request_stop()
    coord.join(threads)



def tryToResize(eval=False):
  dir = Path(data_dirDigitsTrain)

  sess = tf.InteractiveSession()
  #pathDataDir = Path(data_dir, 'train')
  #pathDataDir = Path(data_dirDigitsTrain)

  filenames = list(dir.glob('*.png'))
  # converte in a list of string paths
  #filenames = list(map(lambda x: str(x.absolute()), filenames))
  filenames = ["C:\\tmp\\svhn_dataDigits\\29586_1.png" for x in range(10)] #it goes in error!
  # Create a queue that produces the filenames to read
  # (he converts the strings in tensors) and add them to the fifoqueue
  filename_queue = tf.train.string_input_producer(filenames)
  #string_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(len(filenames)):
    reader = tf.WholeFileReader("reader")
    #restituisce una stringa che rappresenta il contenuto, e una stringa per il filename
    key,value = reader.read(filename_queue, "read")
    img_u = tf.image.decode_jpeg(value, channels=3)
    img_f = tf.cast(img_u, tf.float32)

    img_f = tf.image.random_brightness(img_f, max_delta=60000)  #63
    img_f = tf.image.random_contrast(img_f, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(img_f)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    float_image = tf.image.resize_image_with_pad(img_f, height, width)
    print(i, sess.run([float_image,key]))
  coord.request_stop()
  coord.join(threads)

"""
  #keyexpand = tf.expand_dims(key,0)
  #splitPng = tf.strings.split(keyexpand, "\\")
  #png = splitPng[-1]
  #to fix : 4 -> with dimshape()-1
  #png = tf.sparse_slice(splitPng,[0,4],[1,1])

  img_opsummary = tf.summary.image("img", img_4)

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run([key, pngName, label])) #sta eseguendo lo stesso grafo, key e label corrispondono in questa maniera (un solo run)

    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    #for i in range(1,len(filenames)):
    for i in range(1, 20):
      print(i)
      #print(key.eval())
      img_opsummary = tf.summary.image(str(i), img_4, 1000)
      for i in range(1,15):

        #pngname = key.eval().decode("utf-8").split("\\")[-1]
        imgop_sess = sess.run(img_opsummary) #need label together..
        train_writer.add_summary(imgop_sess)

    coord.request_stop()
    coord.join(threads)
"""





if __name__ == '__main__':
  main()
