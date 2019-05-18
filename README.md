# Using computer vision to detect if bike lanes are blocked
### Project
**2019.05.18**
Dan's most recent pull request update:
"The main scripts are:
* object_detection/py/run_obj_det_models: the main script that sources functions and runs any object detection model you want from TensorFlow or Yolo. The script needs a little adjustment to make it suitable for video data as well, just have to change how the script reads in input data. The needed adjustments can be found in object_detection/py/execute_object_detection_video.ipynb
* image_classification/py/parking_dirty_transfer_learning_with_pretraining: I am giving it another go with transfer learning before we move forward with object detection"

[Parking Dirty](http://parkingdirty.com "Parking Dirty") is a project to crowd-source data on Arlington, VA's frequently-blocked bike lanes.

### Approach
There are three main approaches to extend Parking Dirty
* Training a machine learning model to determine if bike lanes are blocked
* Use pretrained object detection models to detect cars. Create polygons to determine if object bounding boxes overlap with bikelane
* Pulling video directly and using [YOLO](https://pjreddie.com/darknet/yolo/) to determine if bike lanes are blocked

For now, we use [Google Colab](https://colab.research.google.com/notebooks/gpu.ipynb) to train on GPU servers
