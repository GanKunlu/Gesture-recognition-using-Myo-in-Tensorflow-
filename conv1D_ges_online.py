#!/usr/bin/python
'''
Created on 2016/09/27

@author: Gan
'''
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function
import myo
from myo.lowlevel import pose_t, stream_emg
from myo.six import print_
import os
import time
import tensorflow as tf
import numpy as np
import collections
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import threading
import warnings


gesture_Table = ['< fist >', '< finger_spread >', '< move_left >', '< move_right >', '< Three_fingers_grasp>', '< thumb >', '< relax >','Unknown']
gestcount = [0,0,0,0,0,0,0,0] 

scale = np.loadtxt("conv_scale.txt")	

class EMGListener(myo.DeviceListener):

	def __init__(self, t_s = 1/70, queue_size = 40):
		self.emg_deque =  collections.deque(maxlen=queue_size)
		self.t_s = t_s
		self.queue_size = queue_size
		self.start = time.time()
		self.isPrepared = False
		
	def on_connect(self, myo, timestamp):
		print_("Connected to Myo")
		myo.vibrate('short')
		myo.set_stream_emg(stream_emg.enabled)
		myo.request_rssi()
		self.start = time.time()
		
	def on_emg(self, myo, timestamp, emg):
		current = time.time()
		tdiff = current - self.start
		if tdiff > self.t_s:
			self.start = time.time()
			self.emg_deque.append(emg)
			
			if len(self.emg_deque) == self.queue_size and self.isPrepared == False:
				self.isPrepared = True
				
	def on_unlock(self, myo, timestamp):
		print_('unlocked')
		
	def on_lock(self, myo, timestamp):
		print_('locked')

	def on_sync(self, myo, timestamp):
		print_('synced')

	def on_unsync(self, myo, timestamp):
		myo.set_stream_emg(stream_emg.enabled)
		print_('unsynced')
		
	def getEmgData(self):
		if len(self.emg_deque)==self.queue_size:
			return np.array(self.emg_deque)
		else:
			warnings.warn('EMG data is not prepared, just pick an empty EMG data', DeprecationWarning)
			return np.array([0]*(8*self.queue_size))
		
			

def findGesture(gestlist):
	global gesture_Table
	maxcount = 0
	maxgest = ""
	for i,count in enumerate(gestlist):
		if(count > maxcount):
			maxcount = count
			maxgest = gesture_Table[i]
	return maxgest

			
def fit():
	global scale
	global gestcount
	global listener
	global sess
	start_time = time.time()
	[mu,sigma] = np.vsplit(scale,2)
	ges_cut = [0.98,0.99,0.97,0.93,0.98,0.97,0.999]

	while(1):
		if(listener.isPrepared):
			online_EMG = listener.getEmgData()
			normdata = np.abs((online_EMG-mu)/np.sqrt(sigma+0.001))
			input_data = np.reshape(normdata,(1,normdata.shape[0],normdata.shape[1]))
			y_predict = sess.run(y,feed_dict = {x:input_data, keep_prob:1})
			ges_i = int(np.where(y_predict == y_predict.max())[1])
			
			if y_predict.max() > ges_cut[ges_i]:
				rec_gindex = ges_i
			else:
				rec_gindex = -1
			
			gestcount[rec_gindex] += 1
			cur_time = time.time()
			if cur_time - start_time > 0.2:
				gesture =  findGesture(gestcount) 			
				if gesture not in ['Unknown','< relax >']:
					print(gesture)
				#print(y_predict.max())
				gestcount = [0,0,0,0,0,0,0]
				start_time = time.time()	
				

def init_model():  
	global sess
	global x
	global y
	global keep_prob
	sess = tf.InteractiveSession()  

	# Create a multilayer model.
	# Input placeholders
	with tf.name_scope('input'):  
		x = tf.placeholder(tf.float32, [None,40, 8], name='x-input')
		image_shaped_input = tf.reshape(x, [-1, 40, 8, 1])  
		tf.summary.image('input', image_shaped_input, 8)
		keep_prob = tf.placeholder(tf.float32)  
		tf.summary.scalar('dropout_keep_probability', keep_prob)		

		# We can't initialize these variables to 0 - the network will get stuck.  
	def weight_variable(shape):  
		"""Create a weight variable with appropriate initialization."""  
		initial = tf.truncated_normal(shape, stddev=0.1)  
		return tf.Variable(initial)  

	def bias_variable(shape):  
		"""Create a bias variable with appropriate initialization."""  
		initial = tf.constant(0.1, shape=shape)  
		return tf.Variable(initial)  

	def variable_summaries(var, name):  
		"""Attach a lot of summaries to a Tensor."""  
		with tf.name_scope('summaries'):  
			mean = tf.reduce_mean(var)  
			tf.summary.scalar('mean/' + name, mean)  
		with tf.name_scope('stddev'):  
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))  
		tf.summary.scalar('sttdev/' + name, stddev)  
		tf.summary.scalar('max/' + name, tf.reduce_max(var))  
		tf.summary.scalar('min/' + name, tf.reduce_min(var))  
		tf.summary.histogram(name, var)  

	def conv1d_layer(input_tensor, in_channels, out_channels, conv_patch, pool_patch = None, layer_name = None, conv_stride = 1, pool_stride = 1, use_pooling = True):
		"""Reusable code for making a conv1D neural net layer. 

		It compites a 1-D convolution, and then uses relu to nonlinearize. 
		It also sets up name scoping so that the resultant graph is easy to read, and 
		adds a number of summary ops. 
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.		
		with tf.name_scope(layer_name):
		
			with tf.name_scope('weights'):
				weights = weight_variable([conv_patch, in_channels, out_channels]) 				
				variable_summaries(weights, layer_name + '/weights')
				
			with tf.name_scope('biases'):  
				biases = bias_variable([out_channels])  
				variable_summaries(biases, layer_name + '/biases')
				
			with tf.name_scope('conv1d'):  
				h_conv1 = tf.nn.conv1d(input_tensor, weights, stride = conv_stride, padding = 'SAME')
				preactivate = h_conv1 + biases
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
				
			activations = tf.nn.relu(preactivate, 'activation')
			tf.summary.histogram(layer_name + '/activations', activations)
			
			if(use_pooling):
				output = tf.nn.pool(activations, window_shape = [pool_patch], pooling_type = "MAX", strides=[pool_stride], padding='SAME')
			else:
				output = activations
			return output
			
	def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):  
		"""Reusable code for making a simple neural net layer. 

		It does a matrix multiply, bias add, and then uses relu to nonlinearize. 
		It also sets up name scoping so that the resultant graph is easy to read, and 
		adds a number of summary ops. 
		"""  
		# Adding a name scope ensures logical grouping of the layers in the graph.  
		with tf.name_scope(layer_name):  
			# This Variable will hold the state of the weights for the layer  
			with tf.name_scope('weights'):  
				weights = weight_variable([input_dim, output_dim])  
				variable_summaries(weights, layer_name + '/weights')  
			with tf.name_scope('biases'):  
				biases = bias_variable([output_dim])  
				variable_summaries(biases, layer_name + '/biases')  
			with tf.name_scope('Wx_plus_b'):  
				preactivate = tf.matmul(input_tensor, weights) + biases  
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			if act == tf.nn.softmax:
				activations = act(preactivate, -1, 'activation')			
			else:
				activations = act(preactivate, 'activation')  
			tf.summary.histogram(layer_name + '/activations', activations)  
		return activations

	hidden1 = conv1d_layer(x,8,48,3,3,'layer1', 1, 2, use_pooling = True)
	hidden2 = conv1d_layer(hidden1,48,128,3,3,'layer2', 1, 2, use_pooling = True)
	hidden3 = conv1d_layer(hidden2,128,152,3,3,'layer3', 1, None, use_pooling = False)
	hidden4 = conv1d_layer(hidden3,152,152,3,3,'layer4', 1, 2, use_pooling = True)
	
	convs_out =  tf.layers.flatten(hidden4)
	hidden5 = nn_layer(convs_out, 760, 512, 'layer5')  
	hidden6 = nn_layer(hidden5,512,256,'layer6')
	dropped1 = tf.nn.dropout(hidden6, keep_prob) 
	hidden7= nn_layer(dropped1,256,256,'layer7')
	#dropped3 = tf.nn.dropout(hidden7, keep_prob)    
	y = nn_layer(hidden7, 256, 7, 'layer8', act=tf.nn.softmax)
	saver = tf.train.Saver()
	saver.restore(sess, "F:/tensorflow_temp/conv_model.ckpt" )

def main():
	global listener
	init_model()
	myo.init()
	hub = myo.Hub()
	hub.set_locking_policy(myo.locking_policy.none)	
	listener = EMGListener()
	
	fit_thread = threading.Thread(target = fit)
	fit_thread.setDaemon(True)
	fit_thread.start()
	
	hub.run(1000, listener)
	try:
		while hub.running:
			myo.time.sleep(0.2)
	finally:
		hub.shutdown()
	  

if __name__ == '__main__':  
	main() 

	
		



