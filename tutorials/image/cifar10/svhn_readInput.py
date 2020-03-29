from __future__ import print_function
from PIL import Image
import os
import sys
from pathlib import Path, PureWindowsPath
import tensorflow as tf
from six.moves import urllib
import tarfile
import datetime
from pathlib import Path, PureWindowsPath
from tensorflow.python.framework import ops
import random
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

import math

import tensorflow as tf


data_dir = '/tmp/svhn_data'
train_dir = '/tmp/svhn_train'
data_dirDigitsTrain = '/tmp/svhn_dataDigits'
data_dirDigitsEval = '/tmp/svhn_dataDigitsEval'

DATA_URL = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
IMAGE_SIZE = 32 #24

# Global constants describing the CIFAR-10 data set.
BATCH_SIZE = 64 #128, 
NUM_CLASSES = 10 #10 digits
#Esempi in un epoca di train, 50 mila immagini per train e 10 mila per l'eval
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 67814 #Numero esempi per epoca per fare il training
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 67000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 67000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 24000  #Numero esempi per epoca per fare l'eval
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000  #Numero esempi per epoca per fare l'eval

def main():
    print("main readinput")
    #readInfoAndCropDigits()
    #da cambiare e refactoring
    #readInfoAndCropDigitsEval()
    #read_input_train()


def elaborateInput(eval=False):
  """Construct distorted input for SVHN training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  #crop digits if is necessary
  if(eval):
    dir = Path(data_dirDigitsEval)
  else:
    dir = Path(data_dirDigitsTrain)

  if (not dir.exists()):
    if not eval:
      readInfoAndCropDigits()
    else:
      readInfoAndCropDigitsEval()

  #pathDataDir = Path(data_dir, 'train')
  #pathDataDir = Path(data_dirDigitsTrain)

  filenames = list(dir.glob('*.png'))
  # converte in a list of string paths
  filenames = list(map(lambda x: str(x.absolute()), filenames))


  # Create a queue that produces the filenames to read
  # (he converts the strings in tensors) and add them to the fifoqueue
  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('reading'):
      reader = tf.WholeFileReader("reader")
      #restituisce una stringa che rappresenta il contenuto, e una stringa per il filename
      key,value = reader.read(filename_queue, "read")
      img_u = tf.image.decode_jpeg(value, channels=3)
      img_f = tf.cast(img_u, tf.float32)
      #img_4 = tf.expand_dims(img_f, 0)

      #4-D Tensor of shape [batch, height, width, channels] ?? channels = 3 , e altezza e larghezze delle immagini???
      #img_f = tf.image.random_flip_left_right(img_f) # must not used NA
      if not eval:

        """
        rotation:
        degrees = 15
        rateForRadiants = math.pi / 180
    
    
        random_angles = tf.random_uniform([1], minval  = -(degrees * rateForRadiants), maxval=(degrees * rateForRadiants))
    
        # not work: img_f = tf.contrib.image.rotate(img_f, tf.random_uniform([1],(- (degrees * rateForRadiants), (degrees * rateForRadiants))))
        #output = transform(images, angles_to_projective_transforms(angles, image_height, image_width),interpolation=interpolation)
    
        image_height = math_ops.cast(array_ops.shape(img_f)[1], dtypes.float32)[None]
        image_width = math_ops.cast(array_ops.shape(img_f)[2], dtypes.float32)[None]
        img_f = tf.contrib.image.transform(img_f,  tf.contrib.image.angles_to_projective_transforms(random_angles,
        image_height, image_width))
        """
        with tf.name_scope('data_augmentation'):
            img_f = tf.image.random_brightness(img_f, max_delta=63)  #63
            img_f = tf.image.random_contrast(img_f, lower=0.2, upper=1.8)

            # Subtract off the mean and divide by the variance of the pixels.
            float_image = tf.image.per_image_standardization(img_f)
      else:
        float_image = tf.image.per_image_standardization(img_f)
        #float_image = img_f

      height = IMAGE_SIZE
      width = IMAGE_SIZE
      # Set the shapes of tensors.
      #float_image.set_shape([height, width, 3]) #doesnt work!!-> seems just for infos.. it's not our case



      #try a easier way
      splits = tf.string_split([key], "\\")
      pngName = splits.values[-1] #"xxx_label.png"

      #op_printlabel = tf.Print(pngName, [pngName], "tensorLabel")

      label = tf.string_split( [tf.string_split([pngName], "\\.").values[0]] ,'_').values[1]

      #with tf.control_dependencies([op_printlabel]):
      labelNumber = tf.strings.to_number(label, tf.int32)
      float_image = tf.image.resize_image_with_pad(float_image, height, width)

      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4

      if not eval:
          numExample = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
      else:
          numExample = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

      min_queue_examples = int(numExample * min_fraction_of_examples_in_queue)
      print ('Filling queue with %d SVHN images before starting to train. '
               'This will take a few minutes.' % min_queue_examples)

  return generate_image_and_label_batch(float_image, labelNumber,
                                                       min_queue_examples, BATCH_SIZE ,
                                                       shuffle=True)


  # -------------------------------------------------------------------------#-------------------------------------------------------------------------#-------------------------------------------------------------------------#-------------------------------------------------------------------------
  # -------------------------------------------------------------------------#---OTHER STUFF..----------------------------------------------------------------------#-------------------------------------------------------------------------
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
    """


def generate_image_and_label_batch(image, label, min_queue_examples,
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
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def readInfoAndCropDigitsEval():
    pathDataDir = Path(data_dir, 'test')
    listFiles = list(pathDataDir.glob('*.png'))
    #size = (128, 128)

    digitsInfo = read_digitStruct("digitStruct_eval.txt")
    # qui ci occupiamo di tagliare le cifre e fare il dataset

    dir = Path(data_dirDigitsEval)
    if(not dir.exists()):
        dir.mkdir()
    else:
        print("dir digits cropped already exits..")
        return
        #print("cleaning files..")
        #for file in list(dir.glob('*')):
        #    file.unlink()

    iteration = 0
    numberOfImages = len(listFiles)
    apercentage = int(numberOfImages/100)
    print("cropping images:")
    for file in listFiles:
        #file, ext = os.path.splitext(infile)
        fileNamePath = str(file.absolute())
        fileNameImg = fileNamePath.split("\\")[-1]
        infoForDigitsImage = digitsInfo[fileNameImg]

        for singleInfoDigit in infoForDigitsImage:
            im = Image.open(fileNamePath)
            top = singleInfoDigit['top']
            left = singleInfoDigit['left']
            height = singleInfoDigit['height']
            width = singleInfoDigit['width']
            label = singleInfoDigit['label']
            #box – The crop rectangle, as a(left, upper, right, lower) - tuple.
            im = im.crop( (int(left),int(top), int(left)+int(width), int(top)+int(height)) )
            #im = im.resize(size) #thumbnail is better.. -> doesn't work!!!
            #im.thumbnail(size) (never mind, it will do later)..
            fileNameImgForSave = fileNameImg.split(".")[0]+"_" + label + ".png"
            fileToSave = Path(data_dirDigitsEval, fileNameImgForSave)
            im.save(fileToSave, "JPEG")

        iteration +=1
        if(iteration % apercentage == 0):
            print(str(int((iteration/numberOfImages)*100)) + "%", end='...', flush=True)

    if(iteration>=numberOfImages):
        print("100% done")

def readInfoAndCropDigits():
    data_dir = '/tmp/svhn_data'
    pathDataDir = Path(data_dir, 'train')
    listFiles = list(pathDataDir.glob('*.png'))
    size = (128, 128)


    digitsInfo = read_digitStruct("digitStruct_train.txt")
    # qui ci occupiamo di tagliare le cifre e fare il dataset

    dir = Path(data_dirDigitsTrain)
    if(not dir.exists()):
        dir.mkdir()
    else:
        print("dir digits cropped already exits..")
        return
        #print("cleaning files..")
        #for file in list(dir.glob('*')):
        #    file.unlink()

    iteration = 0
    numberOfImages = len(listFiles)
    apercentage = int(numberOfImages/100)
    print("cropping images:")
    for file in listFiles:
        #file, ext = os.path.splitext(infile)
        fileNamePath = str(file.absolute())
        fileNameImg = fileNamePath.split("\\")[-1]
        infoForDigitsImage = digitsInfo[fileNameImg]

        for singleInfoDigit in infoForDigitsImage:
            im = Image.open(fileNamePath)
            top = singleInfoDigit['top']
            left = singleInfoDigit['left']
            height = singleInfoDigit['height']
            width = singleInfoDigit['width']
            label = singleInfoDigit['label']
            #box – The crop rectangle, as a(left, upper, right, lower) - tuple.
            im = im.crop( (int(left),int(top), int(left)+int(width), int(top)+int(height)) )
            #im = im.resize(size) #thumbnail is better.. -> doesn't work!!!
            #im.thumbnail(size) (never mind, it will do later)..
            fileNameImgForSave = fileNameImg.split(".")[0]+"_" + label + ".png"
            fileToSave = Path(data_dirDigitsTrain, fileNameImgForSave)
            im.save(fileToSave, "JPEG")

        iteration +=1
        if(iteration % apercentage == 0):
            print(str(int((iteration/numberOfImages)*100)) + "%", end='...', flush=True)

    if(iteration>=numberOfImages):
        print("100% done")


def read_digitStruct(nomefile):
    with open(nomefile, "r") as f:
        digitDict = {}

        for line in f:
            tokens = line.split(';')
            digitsInfo = {}
            for token in tokens:
                if ':' in token:
                    key = token.split(':')[0].strip(' ')
                    value = token.split(':')[1].strip(' ')
                    if key == "name":
                        if value in digitDict:
                            digitDict[value].append(digitsInfo)
                            #listDigits.append(digitsInfo)
                            #digitDict[value] = listdigits
                        else:
                            digitDict[value] = [digitsInfo] #'1.png': {'top':..., 'left':..., 'height':...,...}
                    else:
                        digitsInfo[key] = value

        return digitDict


if __name__ == '__main__':
  main()
