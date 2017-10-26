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

import nets
import codecs
            
def load_image_file_list(anchor_path, dir_path):
  image_files = []

  # 비교 기준 얼굴 이미지를 맨 앞에 넣는다.
  image_files.insert(0, anchor_path)

  # 이미지 파일 목록을 얻는다.
  exts = ['.jpg', '.png']
  files = glob.glob("{}/**/*".format(dir_path), recursive=True)
  for f in files:
    _, ext = os.path.splitext(f.lower())
    if ext in exts:
      image_files.append(f)

  return image_files


def load_images(files):
  images = [None] * len(image_files)
  for i in range(len(image_files)):
      images[i] = misc.imread(os.path.expanduser(args.image_files[i]))

  return images, image_files

def csv_write(file_name, l):
  with open(file_name, "a") as pf:
    # BOM
    pf.write(u'\ufeff')

    writer = csv.writer(pf)
    for v in l:
      writer.writerow(v)

def grouper(iterable, n, fillvalue=None):
  "Collect data into fixed-length chunks or blocks"
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
  args = [iter(iterable)] * n
  return zip(*args)

# non lazy function
def predict_list(face_recognizer, l):
  file_names, origin_imgs, preprocessed_images = zip(*l)
  bbs, aligneds, croppeds, prewhiteneds = zip(*preprocessed_images)
  embs = face_recognizer.predict(prewhiteneds)
  return zip(file_names, origin_imgs, preprocessed_images, bbs, aligneds, croppeds, embs)


def main(args):
  print(args)
  args.in_dir = os.path.abspath(args.in_dir)
  args.anchor = os.path.abspath(args.anchor)
  print("change dir to abspath")
  print(args)

  # out dir 지우고 새로 만듦
  if os.path.exists(args.out_dir): shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)

  files = load_image_file_list(args.anchor, args.in_dir)
  csv_write(args.out_dir + "/image_files.csv", files)

  net_config = {
          'face_recognizer_model' : args.model,
          'margin' : args.margin,
          'image_size' : args.image_size
          }
  face_recognizer = nets.FaceRecognizer(net_config)

  # Run forward pass to calculate embeddings
  # => [img, img, ...]
  # l = map(misc.imread, files)
  origin_images = map(Image.open, files)

  # rotate 0, 120, 240
  # => [[img, rotated_img, ...], [img, rotated_img, ...], ...]
  rotated_images = itertools.chain.from_iterable(itertools.starmap(lambda f_name, img:  \
                         [(f_name, np.array(img)), \
                          (f_name, np.array(img.copy().rotate(120, expand=True))), \
                          (f_name, np.array(img.copy().rotate(240, expand=True)))],
                       zip(files, origin_images)))

  # list of list : many faces per image
  # => [ [(origin_img, face), ...], [], [], 
  #      ... ]
  l = itertools.chain.from_iterable(itertools.starmap(lambda f_name, img: \
          zip(itertools.repeat(f_name), \
              itertools.repeat(img), \
              itertools.chain.from_iterable(face_recognizer.preprocessing([img]))),
          rotated_images))

  # [ (file_name, origin_img, ...), ... ]
  embed_list = itertools.chain.from_iterable(
          map(partial(predict_list, face_recognizer), grouper(l, 8))) 

  # 결과 저장.
  emb_result = []
  #emb_result.append('file', 'bb[0]', 'bb[1]', 'bb[2]', 'bb[3]')
  (name, origin, preprocessed_image, bb, cropped, aligned, emb) = next(embed_list)
  _, file_name = os.path.split(name)
  anchor_emb = emb
  r = [file_name] + list(itertools.chain.from_iterable([emb, bb, [0]]))
  print(r)
  emb_result.append(r)
  for i, (name, origin, preprocessed_image, bb, cropped, aligned, emb) in enumerate(embed_list):
      _, file_name = os.path.split(name)
      # print('name: {}, bb:{}'.format(name, bb))
      dist = np.sqrt(np.sum(np.square(np.subtract(anchor_emb, emb))))
      #misc.imsave('{}/{:4f}_{}.jpg'.format(aligned_img_dir, dist, i), aligned)
      misc.imsave('{}/dist_{}_{}_{}.jpg'.format(args.out_dir, dist, file_name, i), aligned)
      misc.imsave('{}/aligned_{}_{}.jpg'.format(args.out_dir, file_name, i), aligned)
      misc.imsave('{}/cropped_{}_{}.jpg'.format(args.out_dir, file_name, i), cropped)
      misc.imsave('{}/origin_{}_{}.jpg'.format(args.out_dir, file_name, i), origin)

      r = [file_name] + list(itertools.chain.from_iterable([emb, bb, [dist]]))
      emb_result.append(r)

  csv_write(args.out_dir + "/emb_result.csv", emb_result)


            

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

