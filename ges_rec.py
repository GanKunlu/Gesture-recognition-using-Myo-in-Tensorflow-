'''
Created on 2016/09/27

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
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')  
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')  
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')  
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')  
flags.DEFINE_string('summaries_dir', '/tmp/ges_rec_logs', 'Summaries directory')  


def LoadFromMFolder( Name_list, Data_Columns, start_flag, end_flag, seg_size = 60,\
	path = os.getcwd(), fileSort_key = lambda x:int(x[:-4]), del_sign = '[]', separator = ', '):
	'''
	Load data from multi-folders, every folder must contain several sampledata file with different class like:
		foldername:
			1.file
			2.file
			...
	
	Name_list: The dictionary of foldernames
	fileSort_key: The sort method of everyfolders' file, defaults is sorting without the last 4 Char: '.***'
	path: The path of these folders, defaults is current directory
	'''
	samples_data = []
	samples_labels = []
	for folder_index, name in enumerate(Name_list):
		file_path = path +  "/data" + name + "/"
		file_list = os.listdir(file_path)
		class_num = len(file_list)
		file_list.sort(key = fileSort_key) 
		print('\nload folder:', name)
		for class_index, gfile_name in enumerate(file_list):
			fr = open(file_path+gfile_name,'r')
			arrayOLines = fr.readlines()
			numberOfLines = len(arrayOLines)
			Rawdata_Mat = np.zeros((numberOfLines,Data_Columns))
			
			for line_index, line in  enumerate(arrayOLines):
				line = line.strip()
				line = line.strip(del_sign)
				listFromLine = line.split(separator)
				Rawdata_Mat[line_index,:] = listFromLine[0:Data_Columns]

			gStart = start_flag[folder_index,class_index]
			gEnd = end_flag[folder_index,class_index]
			Prune_mat = Rawdata_Mat[gStart:gEnd:,:]
			Norm_value = len(Prune_mat) % seg_size
        		seg_number = (len(Prune_mat) - Norm_value) // seg_size
			if Norm_value != 0:
        			step_data = Prune_mat[:-Norm_value:1, :].ravel()
			else:
				step_data = Prune_mat.ravel()
        		samples_data.extend(np.hsplit(step_data, seg_number))
			init_label = [0]*class_num
			init_label[class_index] = 1
			samples_labels.extend([np.array(init_label)]*seg_number)
			print(' ', gfile_name,end='')	
	print('\nload done!')
	return samples_data,samples_labels

class DataSet(object):
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
			fake_image = [1] * 480
			fake_label = [1] + [0] * 7

			return [fake_image for _ in xrange(batch_size)], [
			fake_label for _ in xrange(batch_size) ]
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
  
def train():  
	# Import data  
 	name_list = ['/0308gan','/0318gan']
	start_flag = np.array([[110,166,18,67,693,280,538,1770], [77,128,219,275,308,410,194,0]])
	end_flag = np.array([[12316,12092,12327,12751,12209,12335,12483,12054],\
			[11801,11643,13874,13846,13982,13359,13833,11193]])
	data,labels = LoadFromMFolder(name_list, 8, start_flag, end_flag)
	data_train, data_test, labels_train, labels_test = train_test_split(np.abs(data),np.abs(labels))
	train_batch = DataSet(data_train, labels_train)

	sess = tf.InteractiveSession()  
  
	# Create a multilayer model.    
	# Input placehoolders  
	with tf.name_scope('input'):  
		x = tf.placeholder(tf.float32, [None, 480], name='x-input') 
		image_shaped_input = tf.reshape(x, [-1, 60, 8, 1])  
		tf.image_summary('input', image_shaped_input, 8)  
		y_ = tf.placeholder(tf.float32, [None, 8], name='y-input')
		keep_prob = tf.placeholder(tf.float32)  
		tf.scalar_summary('dropout_keep_probability', keep_prob)  
  
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
			tf.scalar_summary('mean/' + name, mean)  
		with tf.name_scope('stddev'):  
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))  
		tf.scalar_summary('sttdev/' + name, stddev)  
		tf.scalar_summary('max/' + name, tf.reduce_max(var))  
		tf.scalar_summary('min/' + name, tf.reduce_min(var))  
		tf.histogram_summary(name, var)  
  
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
				tf.histogram_summary(layer_name + '/pre_activations', preactivate)  
			activations = act(preactivate, 'activation')  
			tf.histogram_summary(layer_name + '/activations', activations)  
      		return activations  
  
	hidden1 = nn_layer(x, 480, 240, 'layer1')  
	hidden2 = nn_layer(hidden1,240,120,'layer2')
	hidden3 = nn_layer(hidden2,120,100,'layer3')
	hidden4 = nn_layer(hidden3,100,80,'layer4')
	dropped2 = tf.nn.dropout(hidden4, keep_prob) 
	hidden5 = nn_layer(dropped2,80,60,'layer5')
	hidden6 = nn_layer(hidden5,60,40,'layer6')
	hidden7= nn_layer(hidden6,40,20,'layer7')
	dropped3 = tf.nn.dropout(hidden7, keep_prob)    
  
	y = nn_layer(dropped3, 20, 8, 'layer8', act=tf.nn.softmax)  
	with tf.name_scope('cross_entropy'):  
		#diff = y_ * tf.log(y) 
		diff = y_ * tf.log(y+1e-10)		
		with tf.name_scope('total'):  
			cross_entropy = -tf.reduce_mean(diff)  
		tf.scalar_summary('cross entropy', cross_entropy)  
  
	with tf.name_scope('train'):  
		train_step = tf.train.AdamOptimizer(  
			FLAGS.learning_rate).minimize(cross_entropy)  
  
	with tf.name_scope('accuracy'):  
		with tf.name_scope('correct_prediction'):  
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  
		with tf.name_scope('accuracy'):  
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
		tf.scalar_summary('accuracy', accuracy)  
  
	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)  
	merged = tf.merge_all_summaries()  
	train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)  
	test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')  
	tf.initialize_all_variables().run()  
  
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
 
def main(_):  
	if tf.gfile.Exists(FLAGS.summaries_dir):  
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)  
	tf.gfile.MakeDirs(FLAGS.summaries_dir)  
	train()  
  
if __name__ == '__main__':  
	tf.app.run()  

	
		



