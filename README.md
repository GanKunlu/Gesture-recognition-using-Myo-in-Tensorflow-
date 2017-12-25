# Gesture-recognition-using-Myo-in-Tensorflow
* Use [myo](https://www.myo.com/) armband to collect multi-sEMG for every gesture by the Linux SDK: [PyoConnect](http://www.fernandocosentino.net/pyoconnect/).
* Gesture recognition by `DNN` in Google's [Tensorflow](https://www.tensorflow.org/) framework.
* The master branch is for the TensorFlow version 1.X(1.0-1.2), and there are another branch for the old version 0.X.
* Add 1D-CNN model as the gesture recohnition algrithm, and achieve online recognition.

## linux: ges_rec.py
* You can visualize the Neural Networks using tensorboard in linux after running ges_rec.py
```
tensorboard --logdir=/tmp/ges_rec_logs/
```
## Windows: 
### conv1D_ges_rec.py  -- model training by 1D-CNN
### conv1D_ges_online.py  -- online recognition by 1D-CNN
* the code is for windows using [myo-python](https://github.com/NiklasRosenstein/myo-python) sdk 
* You can visualize the Neural Networks using tensorboard in windows after running conv1D_ges_rec.py
```
tensorboard --logdir=F:/tensorflow_temp/ges_rec_logs/
```

## requirement
### Linux:
* PyoConnect
* Numpy
* [Scikit-learn](http://scikit-learn.org/stable/index.html)
* Tensorflow
### Windows:
* myo-python
* Numpy
* [Scikit-learn](http://scikit-learn.org/stable/index.html)
* Tensorflow
