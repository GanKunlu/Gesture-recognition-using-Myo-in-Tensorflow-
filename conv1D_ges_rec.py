'''
Created on 2017/09/27

@author: Gan
'''
from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
 
flags = tf.app.flags  
FLAGS = flags.FLAGS  
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '  
                     'for unit testing.')  
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.') 
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')  
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')  
flags.DEFINE_string('data_dir', 'F:/tensorflow_temp/data', 'Directory for storing data')  
flags.DEFINE_string('summaries_dir', 'F:/tensorflow_temp/ges_rec_logs', 'Summaries directory')  


def loadFromMFolder( Name_list, Data_Columns,  seg_size = 40,\
		path = os.getcwd(), fileSort_key = lambda x:int(x[:-4]), del_sign = '[]', separator = ', '):
	'''
	Load data from multi-folders, every folder must contain several sampledata file with different class like:
		foldername:
			1.file
			2.file
			...
	
	Name_list: The dictionary of folder names
	fileSort_key: The sort method of every folders' file, defaults is sorting without the last 4 Char: '.***'
	path: The path of these folders, defaults is current directory
	'''
	samples_data = []
	samples_labels = []
	for person_index, name in enumerate(Name_list):
		file_path = path +  "/data" + name + "/"
		files_list = os.listdir(file_path)
		for files_i, fname in enumerate(files_list):
			curpath = file_path + fname + "/"
			file_list = os.listdir(curpath)
			class_num = len(file_list)
			file_list.sort(key = fileSort_key) 
			print('\nload folder:', fname)
			for class_index, gfile_name in enumerate(file_list):
				fr = open(curpath + gfile_name,'r')
				arrayOLines = fr.readlines()
				numberOfLines = len(arrayOLines)
				Rawdata_Mat = np.zeros((numberOfLines,Data_Columns))

				for line_index, line in  enumerate(arrayOLines):
					line = line.strip()
					line = line.strip(del_sign)
					listFromLine = line.split(separator)
					Rawdata_Mat[line_index,:] = listFromLine[0:Data_Columns]

				Prune_mat = Rawdata_Mat[:,:]
				Norm_value = len(Prune_mat) % seg_size
				seg_number = (len(Prune_mat) - Norm_value) // seg_size
				if Norm_value != 0:
					step_data = Prune_mat[:-Norm_value:1, :]
				else:
					step_data = Prune_mat				
				samples_data.extend(np.vsplit(step_data, seg_number))
				init_label = [0]*class_num
				init_label[class_index] = 1
				samples_labels.extend([np.array(init_label)]*seg_number)
				print(' ', gfile_name,end='')	
	print('\nload done!')
	return samples_data,samples_labels


class DataSet(object):
	"""dataset of EMG data, using mini batch to prepare the input data 
	"""
	def __init__(self, images, labels, fake_data=False):
		if fake_data:
			self._num_examples = 10000
		else:
			assert images.shape[0] == labels.shape[0], (
		'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
			self._num_examples = images.shape[0]
		self._images = np.abs(images)
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 320
			fake_label = [1] + [0] * 6

			return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size) ]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

def normlization(data):
	"""normlize the data by zero-mean
	"""
	raw, val = data[0].shape
	mu = np.zeros((raw, val))
	sigma = np.zeros((raw, val))
	normdata = []
	eps = 0.0001 # the smoothing factor
	for sample in data:
		mu = mu + sample / len(data)
	for sample in data:
		sigma = sigma + ((sample - mu)**2) / len(data)
	for sample in data:
		normdata.append((sample - mu) / np.sqrt(sigma + eps))
	
	np.savetxt("conv_scale.txt", np.vstack((mu,sigma)) )
	return normdata

def train():  
	# Import data
	name_list = ['/Gan','/QiHuan','/QinHao','/YuHailong','/WeiXiaotong']
	data,labels = loadFromMFolder(name_list, 8)
	print(len(data))
	# normlization
	data = normlization(data)
	# tain_test data split
	data_train, data_test, labels_train, labels_test = train_test_split(np.abs(data),np.abs(labels))
	# init Dataset
	train_batch = DataSet(data_train, labels_train)

	sess = tf.InteractiveSession()  

	# Create a multilayer model.
	# Input placeholders
	with tf.name_scope('input'):  
		x = tf.placeholder(tf.float32, [None,40, 8], name='x-input')
		image_shaped_input = tf.reshape(x, [-1, 40, 8, 1])  
		tf.summary.image('input', image_shaped_input, 8)  
		y_ = tf.placeholder(tf.float32, [None, 7], name='y-input')
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
	
	# multi-convolution layers
	hidden1 = conv1d_layer(x,8,48,3,3,'layer1', 1, 2, use_pooling = True)
	hidden2 = conv1d_layer(hidden1,48,128,3,3,'layer2', 1, 2, use_pooling = True)
	hidden3 = conv1d_layer(hidden2,128,152,3,3,'layer3', 1, None, use_pooling = False)
	hidden4 = conv1d_layer(hidden3,152,152,3,3,'layer4', 1, 2, use_pooling = True)
	# flat to vector
	convs_out =  tf.layers.flatten(hidden4)
	# muti_nn layers
	hidden5 = nn_layer(convs_out, 760, 512, 'layer5')  
	hidden6 = nn_layer(hidden5,512,256,'layer6')
	dropped1 = tf.nn.dropout(hidden6, keep_prob) 
	hidden7= nn_layer(dropped1,256,256,'layer7')
	#dropped3 = tf.nn.dropout(hidden7, keep_prob)    
	y = nn_layer(hidden7, 256, 7, 'layer8', act=tf.nn.softmax)  

	with tf.name_scope('cross_entropy'):  
		# diff = y_ * tf.log(y)
		diff = y_ * tf.log(y+1e-10)		
		with tf.name_scope('total'):  
			cross_entropy = -tf.reduce_mean(diff)  
		tf.summary.scalar('cross entropy', cross_entropy)  

	with tf.name_scope('train'):  
		train_step = tf.train.AdamOptimizer(  
			FLAGS.learning_rate).minimize(cross_entropy)  

	with tf.name_scope('accuracy'):  
		with tf.name_scope('correct_prediction'):  
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
		with tf.name_scope('accuracy'):  
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
		tf.summary.scalar('accuracy', accuracy)  

	# Merge all the summaries and write them out to F:/tensorflow_temp/ges_rec_logs (by default)  
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')   

	tf.global_variables_initializer().run()  

	# Train the model, and also write summaries.  
	# Every 10th step, measure test-set accuracy, and write test summaries  
	# All other steps, run train_step on training data, & add training summaries  

	def feed_dict(train):  
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""  
		if train or FLAGS.fake_data:  
			xs, ys = train_batch.next_batch(80, fake_data=FLAGS.fake_data)  
			k = FLAGS.dropout  
		else:  
			xs, ys = data_test, labels_test
			k = 1.0  
		return {x: xs, y_: ys, keep_prob: k}  

	for i in range(FLAGS.max_steps):  
		if i % 10 == 0:  # Record summaries and test-set accuracy  
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))  
			test_writer.add_summary(summary, i)  
			print('Accuracy at step %s: %s' % (i, acc))  
		else: # Record train set summarieis, and train  
			summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, i)

	saver = tf.train.Saver()
	save_path = saver.save(sess,"F:/tensorflow_temp/conv_model.ckpt")


def main(_):  
	if tf.gfile.Exists(FLAGS.summaries_dir):  
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)  
	tf.gfile.MakeDirs(FLAGS.summaries_dir)  
	train()  

if __name__ == '__main__':  
	tf.app.run()  

	
		



