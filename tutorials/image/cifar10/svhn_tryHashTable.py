import numpy as np
import tensorflow as tf

def main(argv=None):
  anotherstuff()
  input_tensor = tf.constant("1.png", dtype=tf.string)
  keys = tf.constant(np.array(["1.png", "2.png","3.png"]), dtype=tf.string)
  #values = tf.constant(np.array([4, 1], [5, 1], [6, 1]), dtype=tf.int64)

  value1 = tf.constant([1,2], dtype=tf.int64)
  #values = tf.constant(np.array([value1,value1,value1]), dtype=tf.int64) -> doens't work (work just with (np.array[1,2,3])..
  #we need a list.. ,and adds elementt ot the list:
  a = np.array([0])
  b = np.array([1])
  c = np.array([1])
  #c2 = np.array([2, 3, 4, 5, 6])
  d = np.concatenate((a,b,c,np.array([2, 3, 4, 5, 6])))
  e = np.concatenate((d,[2]))


  #values = np.array([0,1,2])
  values = np.array(["provaaa","try","try2"])
  #aggiungere i valori ok, ora.. ci serve.. che..
  #values = np.array([[2,3,4],[5,6,7],[7,8,9]])

  #https: // stackoverflow.com / questions / 50315932 / tensorflow - hashtable - lookup -with-arrays
  #purtroppo lavora solo con degli scalari.. quindi??!

  #table = tf.contrib.lookup.HashTable(
  #  tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
  #  -999)
#  #[{'label': '3', 'top': '7', 'left': '52', 'width': '21', 'height': '46'}, {'label': '1', 'top': '10', 'left': '74', 'width': '15', 'height': '46'}]
#Quindi potremmo mettere queste informazioni in stringa.. e poi trattare le stringhe splittando!..

  table = tf.contrib.lookup.HashTable(
   tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
    "default")

  out = table.lookup(input_tensor)


  with tf.Session() as sess:
    table.init.run()
    print(out.eval())


def anotherstuff():
  #what does it do?
  lookup = tf.placeholder(shape=(2,), dtype=tf.int64)
  default_value = tf.constant([1, 1], dtype=tf.int64)
  input_tensor = tf.constant([1, 1], dtype=tf.int64)
  keys = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.int64)
  values = tf.constant([[4, 1], [5, 1], [6, 1]], dtype=tf.int64)
  val0 = values[:, 0] #(4,5,6)
  val1 = values[:, 1] #(1,1,1]

  st0 = tf.SparseTensor(keys, val0, dense_shape=(7, 7)) #print(sess.run(st0))
  '''SparseTensorValue(indices=array([[1, 2],
                                   [3, 4],
                                   [5, 6]], dtype=int64), values=array([4, 5, 6], dtype=int64),
                    dense_shape=array([7, 7], dtype=int64))'''

  st1 = tf.SparseTensor(keys, val1, dense_shape=(7, 7))
  '''SparseTensorValue(indices=array([[1, 2],
                                   [3, 4],
                                   [5, 6]], dtype=int64), values=array([1, 1, 1], dtype=int64),
                    dense_shape=array([7, 7], dtype=int64))'''

  x0 = tf.sparse_slice(st0, lookup, [1, 1])
  y0 = tf.reshape(tf.sparse_tensor_to_dense(x0, default_value=default_value[0]), ())
  x1 = tf.sparse_slice(st1, lookup, [1, 1])
  y1 = tf.reshape(tf.sparse_tensor_to_dense(x1, default_value=default_value[1]), ())
  #NOTA: la separazione delle immagine da fare con tool open cv (preparare in un altra data set).

  y = tf.stack([y0, y1], axis=0)
  #mi sembra di capire che ha messo in stack gli indici con il valore del primo, più gli indici con il secondo valore.. hmm
  #y0-> è st0 con il lookup  più il valore di default, così anche y1, se unisci... hmm


  with tf.Session() as sess:
    #st0.eval()?..
    #print(sess.run(st0))
    #print(sess.run(st1))

    print(sess.run(y, feed_dict={lookup: [1, 2]}))
    print(sess.run(y, feed_dict={lookup: [1, 1]}))
    print(sess.run(y, feed_dict={lookup: [3, 4]}))
    print(sess.run(y, feed_dict={lookup: [5, 6]}))
  #questo funziona... ma comunque non vogliamo il feeddict, penso proprio che non sia un problema...
  #vogliamo che il grafo, la computazione sia a stabilre il lookup, non un placeholder..
  #da capire  questo pezzo di codice..



if __name__ == '__main__':
  tf.app.run() #main()
