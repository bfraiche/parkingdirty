**2019.02.09**

Have a look at [Dan Bernstein](https://github.com/danbernstein "Dan Bernstein Github")'s object detection in the [object detection branch](https://github.com/bfraiche/parkingdirty/tree/object_detection "Object Detection")

I am keeping pull requests there for now

"I used the TensorFlow object detection API to detect all objects in traffic images, then used two methods to classify the objects of whether or not they are bike lane obstacles, one is to use the center point of the object, and the other is to measure how much of the object overlaps with the bike lane. The bike lane is just a polygon I drew on, so it is specific to this traffic camera at this time. agreement between this classification and the parking dirty manual classification is generally good."
