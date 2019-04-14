import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib.path as mpltPath


from PIL import Image
import scipy.misc

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import time
import csv
from datetime import datetime
import datetime

import cv2
import subprocess
import shutil
import matplotlib.path as mpltPath

from google.colab.patches import cv2_imshow 

from PIL import ImageOps
import imageio
from PIL import ImageFont, ImageDraw, Image


import subprocess # for running R script in python
from shapely.geometry import Polygon # to measure overlap between bike lane and bounding boxes
from pascal_voc_writer import Writer

if not os.path.exists('object_detection/input_imgs'):
    os.makedirs('object_detection/input_imgs')
    
    # download parking dirty images here when needed

if not os.path.exists('object_detection/output_imgs'):
    os.makedirs('object_detection/output_imgs')

if not os.path.exists('object_detection/output_csv'):
    os.makedirs('object_detection/output_csv')


def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids.any in ['dog', 'person']:
          print("dog")
        else:
          print("no dog")
          #  car_boxes.append(box)

    return np.array(car_boxes)
  
  
def visualize_boxes(image, box_val, class_val, score_val, threshold):
  vis_util.visualize_boxes_and_labels_on_image_array(
   image,
   np.squeeze(box_val),
   np.squeeze(class_val).astype(np.int32),
   np.squeeze(score_val),
   category_index,
   min_score_thresh=threshold,
   use_normalized_coordinates=True,
   line_thickness=8)

"""piece of code that represent the concrete detection, calling the TF session"""

def process_frames(image_tensor, image_np_expanded, w, h, sess, detection_boxes, detection_scores, detection_classes, num_detections, blocked, ind, threshold, lane_poly):
  for i in image_np_expanded:   
  # Actual detection.
    (boxes, scores, classes, num) = sess.run(
     [detection_boxes, detection_scores, detection_classes, num_detections],
     feed_dict={image_tensor: image_np_expanded}) # Visualization of the results of a detection.

    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)

    points_all = []
    
    lane = np.array(lane_poly)
    #lane = np.array([[0,0],[w,0],[w,h],[0,h]])
    pathbikelane = mpltPath.Path(lane)


    for i in range(boxes.shape[0]):
       if scores[i] > threshold:
          box = tuple(boxes[i].tolist())

          classes_int = np.squeeze(classes).astype(np.int32)

          if classes_int[i] in category_index.keys():
            class_name = category_index[classes_int[i]]['name']

          #print(class_name)

          ymin, xmin, ymax, xmax = box

          # the box is given as a fraction of the distance in each dimension of the image
          # so we have to multiple it by the image dimensions to get the center of each box, relative to the rest of the image
          center_x = (((xmax * w) - (xmin * w)) / 2) + (xmin * w) # x dimension of image
          center_y = (((ymax * h) - (ymin * h)) / 2) + (ymin * h) # y dimension of image
          points = [(center_x, center_y)]

          points_all.append(points)
         # print(points_all)


      #    if any(pathbikelane.contains_points(points)):
      #      print(points)
      #      blocked += 1
    
    if any(pathbikelane.contains_points(x) for x in points_all):
      blocked += 1
      
    ind += 1

    blocked_text = round(100 * blocked / ind, 0)
    
    print(blocked, ind)
  
  return boxes, classes, scores, num, blocked, ind, blocked_text, lane


def set_up_detection(sess, detection_graph):
  
  sess.run(tf.global_variables_initializer())

  # Definite input and output Tensors for detection_graph
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  # score: confidence in the class prediction
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  # detected class
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
  return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections

def process_video(video_file, detection_graph, threshold, lane_poly):
  
  
  try:
    cap = cv2.VideoCapture(video_file)
    print(video_file + " successfully loaded")
  except IOError:
    print("Issue opening "+video_file)

  blocked = 0
  ind = 0
  
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  #fps = int(cap.get(cv2.CAP_PROP_FPS))
  #n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # set up output object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

  print('starting processing')


  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

      image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = set_up_detection(sess, detection_graph)

      # configure tf object detection API for boxes, scores, classes, and num of detections
      while(cap.isOpened()):
        ret, image_np = cap.read)

        if not ret:
          break

        image_np_expanded = np.expand_dims(image_np, axis=0)


        boxes, classes, scores, num, blocked, ind, blocked_text, lane = process_frames(image_tensor, image_np_expanded, w, h, sess, detection_boxes, detection_scores, detection_classes, num_detections, blocked, ind, threshold, lane_poly)        

        #print(blocked)

        visualize_boxes(image_np, boxes, classes, scores, threshold)


        img = image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'blocked' + str(blocked_text) + '% of the time',(10,500), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        lane = np.array(lane_poly)
        pathbikelane = mpltPath.Path(lane)
        cv2.polylines(img, pathbikelane,True,(0,255,255))

        # write out video with object labels
        if ret==True:
          out.write(img)

         # cv2_imshow(image_np)

          if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        else:
           break

  print('successfully run')
  cv2.destroyAllWindows()
  
  
  
def get_first_frame(video_file):
 # video_file = 'out-2.avi'  

  try:
    cap = cv2.VideoCapture(video_file)
    print(video_file + " successfully loaded")
  except IOError:
    print("Issue opening "+video_file)


  print('starting processing')


  # configure tf object detection API for boxes, scores, classes, and num of detections
  while(cap.isOpened()):
    ret, image_np = cap.read()

    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    imageio.imwrite('first-frame.jpg', image_np_expanded[0])
    
    print('first frame written out')


    break
