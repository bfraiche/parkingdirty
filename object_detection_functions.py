import numpy as np
import os
import fnmatch
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


import subprocess # for running R script in python
from shapely.geometry import Polygon # to measure overlap between bike lane and bounding boxes
from pascal_voc_writer import Writer

#shutil.rmtree('object_detection/input_imgs')
#shutil.rmtree('object_detection/output_imgs')
#shutil.rmtree('object_detection/output_csv')
#shutil.rmtree('object_detection/output_xml')

# create folders for input and outputs
# output_img will return the input_img with bounding boxes for objects included
# output_csv will have the outcome of the shapely overlap analysis 

if not os.path.exists('object_detection/input_imgs'):
    os.makedirs('object_detection/input_imgs')
    
    # download parking dirty images here when needed

if not os.path.exists('object_detection/output_imgs'):
    os.makedirs('object_detection/output_imgs')

if not os.path.exists('object_detection/output_csv'):
    os.makedirs('object_detection/output_csv')
    
    
if not os.path.exists('object_detection/output_xml'):
    os.makedirs('object_detection/output_xml/xml_files')

"""**Configuration** of the model to use, path to the frozen inference graph and extra config elements for the Object detection API implementation."""

def download_data(cam):
  if cam == "single":
  
# download and read in data
    zip_address = 'http://parkingdirty.com/BlockedBikeLaneTrainingSingleCam.zip'
  else:
    zip_address = 'http://parkingdirty.com/BlockedBikeLaneTrainingFull.zip'
  
    import requests, zipfile, io
    r = requests.get(zip_address)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('object_detection/input_imgs') # extract images from zip to input_imgs folder
    
    print('data downloaded successfully')

# filters the data based on a regex pattern, returns two lists (vectors?) of the files matching in the blocked and not blocked files, respectively
def filter_data(pattern):
  pattern = '*' + pattern + '*'
  blocked = fnmatch.filter(os.listdir('object_detection/input_imgs/blocked'), pattern)
  notblocked = fnmatch.filter(os.listdir('object_detection/input_imgs/notblocked'), pattern)

  files = [blocked, notblocked]

  return files

# use the filter function to then move the matching files into subdirectories to use for analysis
def subset_data(pattern): 
  pattern_path = 'object_detection/input_imgs_subset_' + pattern
  if not os.path.exists(pattern_path):
  #  shutil.rmtree('object_detection/input_imgs_subset')
    os.makedirs(pattern_path + '/blocked')
    os.makedirs(pattern_path + '/notblocked')
  
  for f in filter_data(pattern)[0]:
      shutil.copy('object_detection/input_imgs/blocked/' + f, pattern_path + '/blocked')
  for f in filter_data(pattern)[1]:
      shutil.copy('object_detection/input_imgs/notblocked/' + f, pattern_path + '/notblocked')

# model download process

# model to use : Faster RCNN Inception ResNet V2 built on coco
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# the function takes a tensorflow model as an argument, but the rest of the 
# set up code is specific to the coco dataset, so other pretrained model datasets 
# would need to be tweaked

def set_up_model(model_name):
  MODEL_NAME = model_name

  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

  NUM_CLASSES = 90

  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # set image size for analysis
  IMAGE_SIZE = (12, 8)
  
  print('model set up successfully')
  
  return detection_graph, label_map, categories, category_index


# helper function to load images

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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


def analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
  
  start_time = time.time()
  timestamp = image_path.split(".png")[0]
  img_name = timestamp.split("/")[-1]


  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  try:
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
  except IOError:
    print("Issue opening "+image_path)
    
    
  width, height = image.size

        
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  
  # if the image file name contains "not" then assigned 0, otherwise 1, so 1 is blocked, 0 is notblocked
  if os.path.join(path_images_dir + "/" + image_path).find('not') is not -1:
    img_labels = 0
  else:
    img_labels = 1 

  # Actual detection
  (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded}) 
  
  
  scores = np.squeeze(scores)
  boxes = np.squeeze(boxes)
  
  return timestamp, img_name, img_labels, boxes, scores, classes, num


def analyze_boxes():
  
  global boxes, scores, classes, lane_poly, pathbikelane, f, threshold, timestamp, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015,num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane
        
        for i in range(boxes.shape[0]):
           if scores[i] > threshold:
              box = tuple(boxes[i].tolist())

              classes_int = np.squeeze(classes).astype(np.int32)

              if classes_int[i] in category_index.keys():
                                  class_name = category_index[classes_int[i]]['name']  


              ymin, xmin, ymax, xmax = box

              # the box is given as a fraction of the distance in each dimension of the image
              # so we have to multiple it by the image dimensions to get the center of each box, relative to the rest of the image
              center_x = (((xmax * 352) - (xmin * 352)) / 2) + (xmin * 352) # x dimension of image
              center_y = (((ymax * 288) - (ymin * 288)) / 2) + (ymin * 288) # y dimension of image
              points = [(center_x, center_y)]
              
              # area of the object
              obj_area =  ((xmax * 352) - (xmin * 352)) * ((ymax * 288) - (ymin * 288))
              
              # get the absolute position of the object in the image
              p1 = Polygon([((xmax * 352),(ymax * 288)), ((xmin * 352),(ymax * 288)), ((xmin * 352),(ymin * 288)), ((xmax * 352),(ymin * 288))])
              
              # location of the bike lane
              p2 = Polygon(lane_poly)
              #print(lane_poly)
              
              # get intersection between object and bike lane
              p3 = p1.intersection(p2)
              
              # get ratio of overlap to total object area
              overlap = p3.area / obj_area
              

              #print(class_name)
              if class_name in {'car', 'truck', 'bus', 'motorcycle','train','person'}:
                if overlap >= 0.1:
                    num_cars_in_bikelane_01 += 1
                if overlap >= 0.15:
                    num_cars_in_bikelane_015 += 1
                if overlap >= 0.2:
                    num_cars_in_bikelane_02 += 1
                if overlap >= 0.25:
                    num_cars_in_bikelane_025 += 1
                if overlap >= 0.3:
                    num_cars_in_bikelane_03 += 1
                if overlap >= 0.35:
                    num_cars_in_bikelane_035 += 1
                if overlap >= 0.4:
                    num_cars_in_bikelane_04 += 1
                if overlap >= 0.45:
                    num_cars_in_bikelane_045 += 1
                if overlap >= 0.5:
                    num_cars_in_bikelane_05 += 1    
                if pathbikelane.contains_points(points):
                    num_cars_in_bike_lane_contains +=1
              
              if class_name == 'bicycle':
                if pathbikelane.contains_points(points):
                    num_bikes_in_bike_lane += 1    
                    
 
        f.write(timestamp + ',' + 
                str(num_cars_in_bikelane_01) + ',' +
                str(num_cars_in_bikelane_015) + ',' +
                str(num_cars_in_bikelane_02) + ',' +
                str(num_cars_in_bikelane_025) + ',' +
                str(num_cars_in_bikelane_03) + ',' +
                str(num_cars_in_bikelane_035) + ',' +
                str(num_cars_in_bikelane_04) + ',' +
                str(num_cars_in_bikelane_045) + ',' +
                str(num_cars_in_bikelane_05) + ',' + 
                str(num_cars_in_bike_lane_contains) + ',' + 
                str(num_bikes_in_bike_lane) + ',' + 
                str(img_labels) + '\n')
    
    # return the data table
        return f
    
"""piece of code that represent the concrete detection, calling the TF session"""

def process_images(detection_graph, path_images_dir, save_directory, threshold, n, lane_poly):
  
  csv_file = 'object_detection/output_csv/csvfile.csv'

  f = open(csv_file, 'w')
  
  print('starting processing')
  print(datetime.datetime.now())
  
  
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      
      # configure tf object detection API for boxes, scores, classes, and num of detections
      image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = set_up_detection(sess, detection_graph)
        
        # loop through the object detection algorithm for each image
        
      if n == 'all':  
        # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
        for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files]:
  #       for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files:
         
          timestamp, img_name, img_labels, boxes, scores, classes, num = analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)        
          
          num_cars_in_bikelane_01, num_cars_in_bikelane_015, num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0        
  
          # the lane polygon is specific to each camera at a particular point in time
          # it could change if the camera's perspective is changed
          # a more robust solution would automatically identify bike lanes
          # lane points identified with: https://www.image-map.net/
          lane = np.array(lane_poly)
          pathbikelane = mpltPath.Path(lane)
  
    # analyzing the detected objects for which are in the bikelane and converting into a tabular format 
    #      writer = Writer(image_path, width, height)
  
          analyze_boxes(boxes, scores, classes, lane_poly, pathbikelane, f, threshold, timestamp, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
          num_cars_in_bikelane_02, num_cars_in_bikelane_025, 
          num_cars_in_bikelane_03, num_cars_in_bikelane_035, 
          num_cars_in_bikelane_04, num_cars_in_bikelane_045,
          num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, 
          num_bikes_in_bike_lane)
      else:  
        # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
        for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files[:n]]:
  #       for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files:
         
          timestamp, img_name, img_labels, boxes, scores, classes, num = analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)        
          
          num_cars_in_bikelane_01, num_cars_in_bikelane_015, num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0        
  
          # the lane polygon is specific to each camera at a particular point in time
          # it could change if the camera's perspective is changed
          # a more robust solution would automatically identify bike lanes
          # lane points identified with: https://www.image-map.net/
          lane = np.array(lane_poly)
          pathbikelane = mpltPath.Path(lane)
  
    # analyzing the detected objects for which are in the bikelane and converting into a tabular format 
    #      writer = Writer(image_path, width, height)
  
          analyze_boxes()

          #print("Process Time " + str(time.time() - start_time))
          #scipy.misc.imsave('object_detection/output_imgs/' + os.path.split(image_path)[1], image_np) # save csv to a different directory than annotated images
        
  f.close()
  print('successfully run')
  print(datetime.datetime.now())
  return csv_file

"""# download data, set up the model, and then run the model on a dataset"""

#download_data()

# set up the model
#detection_graph, label_map, categories, category_index = set_up_model('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')


def visualize_boxes(image_path, detection_graph, threshold, lane_poly):

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # configure tf object detection API for boxes, scores, classes, and num of detections
      
      image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = set_up_detection(sess, detection_graph)
      image_np = cv2.imread(image_path)
     # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
      IMAGE_SIZE = (12, 8)

#      image_np = load_image_into_numpy_array(image)
 #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#      image = clahe.apply(image_np)

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})# Visualization of the results of a detection.
      
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=threshold,
          use_normalized_coordinates=True,
          line_thickness=2)
          
      lane = np.array([lane_poly], np.int32)
      overlay = image_np.copy()
      alpha = 0.7
      beta = ( 1.0 - alpha );
      
      src2 = cv2.fillPoly(image_np, lane, (255, 255, 0))
      frame_out = cv2.addWeighted(overlay, alpha, src2, beta, 0, image_np);

      
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(frame_out)
      
      imageio.imwrite('object_detection/output_imgs/' + os.path.split(image_path)[1], frame_out) # save csv to a different directory than annotated images

 
# clone dan bernstein's parkingdirty repo to access the R script for analysis

def get_optimal_threshold(file):

  command = 'Rscript'
  path2script = 'parkingdirty/analyze_output.R'

  args = [file]
  cmd = [command, path2script] + args
  x = subprocess.check_output(cmd, universal_newlines=True)

  print(x)
  
  
def get_misclassification(file, n):

  command = 'Rscript'
  path2script = 'parkingdirty/get_misclassification.R'

  args = [file, n]
  cmd = [command, path2script] + args
  x = subprocess.check_output(cmd, universal_newlines=True)

  print(x)
  
def plot_classification_by_hour(file):

  command = 'Rscript'
  path2script = 'parkingdirty/mis_classification_by_time.R'

  args = [file]
  cmd = [command, path2script] + args
  x = subprocess.check_output(cmd, universal_newlines=True)

  print(x)




