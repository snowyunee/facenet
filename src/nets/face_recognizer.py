import time
import threading
from scipy import misc

import logging
import numpy as np
import tensorflow as tf
import collections
import cv2
import PIL.ImageDraw as ImageDraw

from skimage import transform as trans

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

  def predict(self, preprocessed_images):
    feed_dict = {
            self._facenet['input'] : preprocessed_images,
            self._facenet['phase_train'] :  False
            }
    embs = self._facenet['sess'].run(self._facenet['output'], feed_dict=feed_dict)
    return embs

  def preprocessing(self, images):
    print('------------------- preprocessing')
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_samples = len(images)
    img_list = []
    result = []
    for i in range(nrof_samples):
        #img = misc.imread(os.path.expanduser(image_paths[i]))
        img = images[i]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, landmarks = align.detect_face.detect_face(
                                      img[:,:,0:3], minsize,
                                      self._mtcnn['pnet'],
                                      self._mtcnn['rnet'],
                                      self._mtcnn['onet'],
                                      threshold, factor)
        landmarks = np.transpose(landmarks)
        # print(">> {} bounding_boxes ---------------------".format(i))
        # print(bounding_boxes.shape)
        # print("<< bounding_boxes ---------------------")
        per_image = []
        for bb_idx in range(bounding_boxes.shape[0]):
            det = np.squeeze(bounding_boxes[bb_idx,0:4])
            # print(">> {} det ---------------------".format(bb_idx))
            # print(bb_idx, det)
            # print("<< det ---------------------")
            bb = np.zeros(4, dtype=np.int32)
            margin = self._config['margin']
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],0:3]
            image_size = self._config['image_size']

            # keypoints = landmarks[bb_idx]
            # keypoints = keypoints.reshape(2,5)
            # keypoints = keypoints - np.array([bb[0], bb[1]]).reshape(2,1)
            # keypoints = np.transpose(keypoints)

            # # print('keypoints')
            # # print(keypoints)
            # self._draw_keypoints_on_image_array(cropped, keypoints[0:2],
            #                       radius=5, use_normalized_coordinates=False)

            try:
              aligned = self._align_face(cropped, image_size,
                                         landmarks[bb_idx],
                                         origin = [bb[0], bb[1]])
            except: 
              print("fail to align face")
              continue

            prewhitened = self._prewhiten(aligned)
            # aligned 는 테스트를 위해 일단 넣어둔다.
            per_image.append((bb, cropped, aligned, prewhitened))

        result.append(per_image)

    return result


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


  def _prewhiten(self, x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

  def _align_face(self, img, src_img_size, landmark, origin):
    dst_size = 112
    dst = np.array([
      [30.2946 + 8, 51.6963],
      [65.5318 + 8, 51.5014],
      [48.0252 + 8, 71.7366],
      [33.5493 + 8, 92.3655],
      [62.7299 + 8, 92.2041] ], dtype=np.float32 )
    p = src_img_size / dst_size
    dst = dst * p

    landmark = landmark.reshape(2,5)
    landmark = landmark - np.array(origin).reshape(2,1)
    src = np.transpose(landmark)
    #dst = np.transpose(landmark).reshape(1,5,2)
    #src = src.reshape(1,5,2)
    # print(src)
    # print(dst)
    # transmat = cv2.estimateRigidTransform(dst.astype(np.float32),
    #                                       src.astype(np.float32), False)
    # out = cv2.warpAffine(img, transmat, (dst_img_size, dst_img_size))
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]
    out = cv2.warpAffine(img, M, (src_img_size, src_img_size), borderValue=0.0)
    return out

  def _draw_keypoints_on_image(self, image,
                              keypoints,
                              color='red',
                              radius=2,
                              use_normalized_coordinates=True):
    """Draws keypoints on an image.
  
    Args:
      image: a PIL.Image object.
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    print('image size:' , im_width, ' ', im_height)
    keypoints_x = [k[0] for k in keypoints]
    keypoints_y = [k[1] for k in keypoints]
    if use_normalized_coordinates:
      keypoints_x = tuple([im_width * x for x in keypoints_x])
      keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
      draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                    (keypoint_x + radius, keypoint_y + radius)],
                   outline=color, fill=color)

  def _draw_keypoints_on_image_array(self, image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
    """Draws keypoints on an image (numpy array).

    Args:
      image: a numpy array with shape [height, width, 3].
      keypoints: a numpy array with shape [num_keypoints, 2].
      color: color to draw the keypoints with. Default is red.
      radius: keypoint radius. Default value is 2.
      use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    self._draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


  
