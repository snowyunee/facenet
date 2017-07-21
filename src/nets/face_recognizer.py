import time
import threading
from scipy import misc

import logging
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

import align.detect_face

class FaceRecognizer(object):
  def __init__(self, config):
    self._config = config
    print(config)
    print('load_facenet')
    self._facenet = self._load_facenet(config)
    print('load_ntcnn')
    self._mtcnn   = self._load_mtcnn()
    print('load_complete')

  def predict(self, images):
    images = self._preprocessing(images)
    feed_dict = {
            self._facenet['input'] : images,
            self._facenet['phase_train'] :  False
            }
    output = self._facenet['sess'].run(self._facenet['output'], feed_dict=feed_dict)
    return output


  def _load_facenet(self, config):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = True

    graph_def = tf.GraphDef()
    with open(config['face_recognizer_model'], 'rb') as file:
      graph_def.ParseFromString(file.read())

    facenet = {}
    facenet['graph'] = tf.Graph()
    facenet['sess'] = tf.Session(config=tf_config, graph=facenet['graph'])

    with facenet['sess'].graph.as_default():
      tf.import_graph_def(graph_def, name="")

    input_name = 'input:0'
    embeddings_name = 'embeddings:0'
    phase_train_name = 'phase_train:0'
    facenet['input'] = facenet['sess'].graph.get_tensor_by_name(input_name)
    facenet['output'] = facenet['sess'].graph.get_tensor_by_name(embeddings_name)
    facenet['phase_train'] = facenet['sess'].graph.get_tensor_by_name(phase_train_name)

    return facenet

  def _load_mtcnn(self):
    
    print('Creating networks and loading parameters')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = True
    tf_config.log_device_placement=False

    mtcnn = {}
    mtcnn['graph'] = tf.Graph()
    mtcnn['sess'] = tf.Session(config=tf_config, graph=mtcnn['graph'])

    with mtcnn['sess'].graph.as_default():
      with mtcnn['sess'].as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(mtcnn['sess'], None)
  
    mtcnn['pnet'] = pnet
    mtcnn['rnet'] = rnet
    mtcnn['onet'] = onet
    return mtcnn


  def _preprocessing(self, images):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_samples = len(images)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        #img = misc.imread(os.path.expanduser(image_paths[i]))
        img = images[i]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize,
                                                          self._mtcnn['pnet'],
                                                          self._mtcnn['rnet'],
                                                          self._mtcnn['onet'],
                                                          threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-self._config['margin']/2, 0)
        bb[1] = np.maximum(det[1]-self._config['margin']/2, 0)
        bb[2] = np.minimum(det[2]+self._config['margin']/2, img_size[1])
        bb[3] = np.minimum(det[3]+self._config['margin']/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped,
                               (self._config['image_size'], self._config['image_size']),
                               interp='bilinear')
        prewhitened = self._prewhiten(aligned)
        img_list[i] = prewhitened

    images = np.stack(img_list)

    return images

  def _prewhiten(self, x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

