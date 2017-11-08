"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
from scipy import misc
import numpy as np
import shutil
import glob
import itertools
import csv
from functools import partial
from PIL import Image
from PIL import ImageDraw

import nets
import codecs
            
def _load_image_file_list(anchor_path, dir_path):
  files = glob.glob("{}/**/*".format(dir_path), recursive=True)
  return [anchor_path] + files


def _csv_write(file_name, l):
  with open(file_name, "a") as pf:
    # BOM
    pf.write(u'\ufeff')

    writer = csv.writer(pf)
    for v in l:
      writer.writerow(v)

def _grouper(iterable, n, fillvalue=None):
  "Collect data into fixed-length chunks or blocks"
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
  args = [iter(iterable)] * n
  return itertools.zip_longest(*args)

def _resize_1024(img):
  m = max(img.size[0], img.size[1])
  if m > 1024:
    x = int(float(img.size[0]) * (1024.0/float(m)))
    y = int(float(img.size[1]) * (1024.0/float(m)))
    return img.resize((x, y), resample=Image.LANCZOS)
  return img

def _rotate(image, angle):
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  image_pil = image_pil.rotate(angle, resample=Image.BILINEAR, expand=True)
  return np.array(image_pil)

def _draw_rectangle(image, bbs):
  draw = ImageDraw.Draw(image)
  for bb in bbs:
    draw.rectangle(bb.tolist(), fill='red', outline='red')

def _draw_rectangle_from_array(image, bbs):
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  _draw_rectangle(image_pil, bbs)
  np.copyto(image, np.array(image_pil))

def _open(path):
  try:
    print(path)
    return Image.open(path)
  except:
    return None

# return shape : (len(images), len(angels))
def _preprocessing(face_recognizer, images):
  angles = [0, 120, 120]

  # rotated_images = []
  results = [[[] for _ in range(len(angles)) ] for _ in range(len(images)) ]
  print(results)
  for i, angle in enumerate(angles):

    # rotate
    images = [_rotate(img, angle) for img in images if img is not None]

    # preprocessing
    preprocessed_list = face_recognizer.preprocessing(images)

    # 얼굴 영역 칠함 다음에 rotate 시켜도 중복으로 얼굴이 detect 되지 않도록
    for img, preprocessed in zip(images, preprocessed_list):
      if len(preprocessed) is 0: continue
      bbs, _, _, _ = zip(*preprocessed)
      _draw_rectangle_from_array(img, bbs)

    # 결과 저장
    # rotated_images.append(images)
    for j, p in enumerate(preprocessed_list):
      results[j][i] = p

  return results, np.array([0,120,240]*len(images)).reshape(-1,3)


def main(args):
  print(args)
  args.in_dir = os.path.abspath(args.in_dir)
  args.anchor = os.path.abspath(args.anchor)
  print("change dir to abspath")
  print(args)

  # out dir 지우고 새로 만듦
  if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)

  files = _load_image_file_list(args.anchor, args.in_dir)
  _csv_write(args.out_dir + "/image_files.csv", files)

  net_config = {
          'face_recognizer_model' : args.model,
          'margin' : args.margin,
          'image_size' : args.image_size
          }
  face_recognizer = nets.FaceRecognizer(net_config)


  batch_size = 1

  for file_list in _grouper(files, batch_size):

    origin_images = [_open(f) for f in file_list if f is not None]

    # remove none image files
    tmp = [(f, img) for f, img in zip(file_list, origin_images) if img is not None]
    if len(tmp) is 0: continue
    file_list, origin_images = zip(*tmp)

    # resize
    origin_images = [np.array(_resize_1024(img)) for img in origin_images]

    # rotate and preprocessing
    preprocessed_list, angles = _preprocessing(face_recognizer, origin_images)

    # flatten preprocessed
    l = list(itertools.chain.from_iterable(
        itertools.starmap(lambda f, o, a, p:
          zip(itertools.repeat(f), itertools.repeat(o), a, p),
        zip(file_list, origin_images, angles, preprocessed_list))))

    l = list(itertools.chain.from_iterable(
        itertools.starmap(lambda f, o, a, p:
          zip(itertools.repeat(f), itertools.repeat(o), itertools.repeat(a), p),
        l)))

    if len(l) is 0: continue
    _, _, _, preprocesses = zip(*l)
    if len(preprocesses) is 0: continue
    _, _, _, prewhitened_images = zip(*preprocesses)

    # predict : get embedding
    embs = face_recognizer.predict(prewhitened_images)
  
    anchor_emb = None
    for i, (v, emb) in enumerate(zip(l, embs)):
      f_name, origin, angle, preprocessed = v
      bb, cropped, aligned, prewhitened = preprocessed

      _, file_name = os.path.split(f_name)

      if anchor_emb is None: anchor_emb = emb

      misc.imsave('{}/origin_{}_{}.jpg'.format(args.out_dir, file_name, i), origin)
      print('------- name: {}, bb:{}, img_size:{}'.format(file_name, bb, origin.size))

      dist = np.sqrt(np.sum(np.square(np.subtract(anchor_emb, emb))))
      misc.imsave('{}/dist_{:0.4f}_{}_{}.jpg'.format(args.out_dir, dist, file_name, i), aligned)
      misc.imsave('{}/aligned_{}_{}.jpg'.format(args.out_dir, file_name, i), aligned)
      misc.imsave('{}/cropped_{}_{}.jpg'.format(args.out_dir, file_name, i), cropped)
    

            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--anchor', type=str, help='anchor face')
    parser.add_argument('--in_dir', type=str, help='Directory of images to compare', default='./in')
    parser.add_argument('--out_dir', type=str, help='Output directory for results', default='./out')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=0)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

