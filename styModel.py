#
#
#Morgans great example code:
#https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
#
# GitHub utility for freezing graphs:
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#
#https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants


import tensorflow as tf
import numpy as np
import trainedModel

#global variables 
g_st_saver=None
g_chkptdir=None
g_trainedgraph=None
g_graph=None


k_freqbins=256
k_width=856

VERBOSE=0

#------------------------------------------------------------
K_ConvRows=1
K_ConvCols=5

k_height=1

k_inputChannnels=k_freqbins

k_ConvStrideRows=1
k_ConvStrideCols=1

k_poolRows = 1    # default for freqs as channels
k_poolStride = 1  # default for freqs as channels

k_downsampledHeight = 1			# default for freqs as channels
k_downsampledWidth = k_width/4 # no matter what the orientation - freqs as channels or as y dim

L1_CHANNELS=64
L2_CHANNELS=64
FC_SIZE = 32

k_convLayerOutputChannels = L2_CHANNELS

k_numClasses=2

#-------------------------------------------------------------

def getShape(g, name) :
	return g.get_tensor_by_name(name + ":0").get_shape()

def constructSTModel(tg) :
	global g_graph
	g_graph = {} 


	#This is the variable that we will "train" to match style and content images.
	#WRONG D?????g_graph["x_image"] = tf.Variable(np.zeros([1,1,k_width, k_freqbins]), dtype=tf.float32, name="s_x_image")
	g_graph["X"] = tf.Variable(np.zeros([1,k_width*k_freqbins]), dtype=tf.float32, name="s_x_image")
	g_graph["x_image"] = tf.reshape(g_graph["X"], [1,k_height,k_width,k_inputChannnels])

	g_graph["w1"]=tf.Variable(tf.truncated_normal([K_ConvRows, K_ConvCols, k_inputChannnels, L1_CHANNELS], stddev=0.1), name="s_w1")
	g_graph["b1"]=tf.Variable(tf.constant(0.1, shape=[L1_CHANNELS]), name="s_b1")
	#g_graph["w1"]=tf.Variable(tf.truncated_normal(getShape( tg, "w1"), stddev=0.1), name="w1")
	#g_graph["b1"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b1")), name="b1")
	
	#             tf.nn.relu(tf.nn.conv2d(x_image,            w1,            strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + b1,            name="h1")
	g_graph["h1"]=tf.nn.relu(tf.nn.conv2d(g_graph["x_image"], g_graph["w1"], strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + g_graph["b1"], name="s_h1")
	# 2x2 max pooling
	g_graph["h1pooled"] = tf.nn.max_pool(g_graph["h1"], ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStride, 2, 1], padding='SAME', name="s_h1_pooled")

	g_graph["w2"]=tf.Variable(tf.truncated_normal([K_ConvRows, K_ConvCols, L1_CHANNELS, L2_CHANNELS], stddev=0.1), name="s_w2")
	g_graph["b2"]=tf.Variable(tf.constant(0.1, shape=[L2_CHANNELS]), name="s_b2")
	#g_graph["w2"]=tf.Variable(tf.truncated_normal(getShape( tg, "w2"), stddev=0.1), name="w2")
	#g_graph["b2"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b2")), name="b2")

	g_graph["h2"]=tf.nn.relu(tf.nn.conv2d(g_graph["h1pooled"], g_graph["w2"], strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + g_graph["b2"], name="s_h2")

	g_graph["h2pooled"] = tf.nn.max_pool(g_graph["h2"], ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStride, 2, 1], padding='SAME', name='s_h2_pooled')
	g_graph["convlayers_output"] = tf.reshape(g_graph["h2pooled"], [-1, k_downsampledWidth * k_downsampledHeight*L2_CHANNELS]) # to prepare it for multiplication by W_fc1

#
	g_graph["W_fc1"] = tf.Variable(tf.truncated_normal([k_downsampledWidth * k_downsampledHeight * k_convLayerOutputChannels, FC_SIZE], stddev=0.1), name="s_W_fc1")
	g_graph["b_fc1"] = tf.Variable(tf.constant(0.1, shape=[FC_SIZE]) , name="s_b_fc1")

	#g_graph["keepProb"]=tf.placeholder(tf.float32, (), name= "keepProb")
	#g_graph["h_fc1"] = tf.nn.relu(tf.matmul(tf.nn.dropout(g_graph["convlayers_output"], g_graph["keepProb"]), g_graph["W_fc1"]) + g_graph["b_fc1"], name="h_fc1")
	g_graph["h_fc1"] = tf.nn.relu(tf.matmul(g_graph["convlayers_output"], g_graph["W_fc1"]) + g_graph["b_fc1"], name="s_h_fc1")


	#Read out layer
	g_graph["W_fc2"] = tf.Variable(tf.truncated_normal([FC_SIZE, k_numClasses], stddev=0.1), name="s_W_fc2")
	g_graph["b_fc2"] = tf.Variable(tf.constant(0.1, shape=[k_numClasses]), name="s_b_fc2")


	g_graph["logits_"] = tf.matmul(g_graph["h_fc1"], g_graph["W_fc2"])
	g_graph["logits"] = tf.add(g_graph["logits_"] , g_graph["b_fc2"] , name="s_logits")


	g_graph["softmax_preds"] = tf.nn.softmax(logits=g_graph["logits"], name="s_softmax_preds")

	return g_graph



def load(meta_model_file, restore_chkptDir) :

	global g_st_saver
	global g_chkptdir
	global g_trainedgraph

	g_chkptdir=restore_chkptDir # save in global for use during initialize

	g_trainedgraph, g_st_saver = trainedModel.load(meta_model_file, restore_chkptDir)
	g = constructSTModel(g_trainedgraph)

	return g

def initialize_variables(sess) :
	global g_graph

	#First initalize all variables
	sess.run ( tf.global_variables_initializer ())
	#next restore the trained graph variable values
	g_st_saver.restore(sess, tf.train.latest_checkpoint(g_chkptdir))

	tf.GraphKeys.USEFUL = 'useful'
	var_list = tf.get_collection(tf.GraphKeys.USEFUL)

	#print('var_list[3] is ' + str(var_list[3]))

	# Now get the values of the trained graph in to the new style graph
	sess.run(g_graph["w1"].assign(g_trainedgraph.get_tensor_by_name("w1:0")))
	sess.run(g_graph["b1"].assign(g_trainedgraph.get_tensor_by_name("b1:0")))
	sess.run(g_graph["w2"].assign(g_trainedgraph.get_tensor_by_name("w2:0")))
	sess.run(g_graph["b2"].assign(g_trainedgraph.get_tensor_by_name("b2:0")))

	sess.run(g_graph["W_fc1"].assign(g_trainedgraph.get_tensor_by_name("W_fc1:0")))
	sess.run(g_graph["b_fc1"].assign(g_trainedgraph.get_tensor_by_name("b_fc1:0")))
	sess.run(g_graph["W_fc2"].assign(g_trainedgraph.get_tensor_by_name("W_fc2:0")))
	sess.run(g_graph["b_fc2"].assign(g_trainedgraph.get_tensor_by_name("b_fc2:0")))

	#sess.run(g_graph["w1"].assign(var_list[3]))
	#sess.run(g_graph["b1"].assign(var_list[4]))
	#sess.run(g_graph["w2"].assign(var_list[5]))
	#sess.run(g_graph["b2"].assign(var_list[6]))

	#sess.run(g_graph["W_fc1"].assign(var_list[7]))
	#sess.run(g_graph["b_fc1"].assign(var_list[8]))
	#sess.run(g_graph["W_fc2"].assign(var_list[9]))
	#sess.run(g_graph["b_fc2"].assign(var_list[10]))

	if VERBOSE : 
		print ('...GLOBAL_VARIABLES :')  #probalby have to restore from checkpoint first
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)



