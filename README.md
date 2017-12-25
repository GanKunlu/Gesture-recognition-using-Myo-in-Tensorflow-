# Gesture-recognition-using-Myo-in-Tensorflow
* Use [myo](https://www.myo.com/) armband to collect multi-sEMG for every gesture by the Linux SDK: [PyoConnect](http://www.fernandocosentino.net/pyoconnect/).
* Gesture recognition by `DNN` in Google's [Tensorflow](https://www.tensorflow.org/) framework.
* The master branch is for the TensorFlow version 1.X(1.0-1.2), and there are another branch for the old version 0.X.
* Add 1D-CNN model as the gesture recohnition algrithm, and achieve online recognition.
* You can visualize the Neural Networks using tensorboard：
```
tensorboard --logdir=/tmp/ges_rec_logs/
```
## requirement
* PyoConnect
* Numpy
* [Scikit-learn](http://scikit-learn.org/stable/index.html)
* Tensorflow

