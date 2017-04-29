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

from PIL import TiffImagePlugin, ImageOps
from PIL import Image


import pickle

g_graph=None

k_freqbins=257
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

L1_CHANNELS=32
L2_CHANNELS=64
FC_SIZE = 32

k_convLayerOutputChannels = L2_CHANNELS

k_numClasses=2

#-------------------------------------------------------------

def getShape(g, name) :
	return g.get_tensor_by_name(name + ":0").get_shape()

def loadImage(fname) :
	#transform into 1D width with frequbins in channel dimension (we do this in the graph in the training net, but not with this reconstructed net)
	return np.transpose(np.reshape(np.array(Image.open(fname).point(lambda i: i*255)), [1,k_freqbins,k_width,1]), [0,3,2,1]) 

def generate_noise_image(content_image, height, width, channels, noise_ratio=0.6):
    noise_image = np.random.uniform(-1, 1, 
                                    (1, height, width, channels)).astype(np.float32)
    print('noise_image shape is ' + str(noise_image.shape))
    return noise_image * noise_ratio + content_image * (1. - noise_ratio)

def save_image(image, fname, scaleinfo=None):
	print('save_image: shape is ' + str(image.shape))
	print('image max is ' + str(np.amax(image) ))
	print('image min is ' + str(np.amin(image) ))
	# Output should add back the mean pixels we subtracted at the beginning
	image = np.clip(image, 0, 255).astype('uint8')

	info = TiffImagePlugin.ImageFileDirectory()
    
	if (scaleinfo == None) :
	    info[270] = '80, 0'
	else :
	    info[270] = scaleinfo

	#scipy.misc.imsave(path, image)

	bwarray=np.asarray(image)/255.

	savimg = Image.fromarray(np.float64(bwarray)) #==============================
	savimg.save(fname, tiffinfo=info)
	#print('RGB2TiffGray : tiffinfo is ' + str(info))
	return info[270] # just in case you want it for some reason
    

def constructSTModel(state) :
	global g_graph
	g_graph = {} 


	#This is the variable that we will "train" to match style and content images.
	##g_graph["X"] = tf.Variable(np.zeros([1,k_width*k_freqbins]), dtype=tf.float32, name="s_x_image")
	##g_graph["x_image"] = tf.reshape(g_graph["X"], [1,k_height,k_width,k_inputChannnels])

	g_graph["X"] = tf.Variable(np.zeros([1,k_height,k_width,k_inputChannnels]), dtype=tf.float32, name="s_X")
	
	g_graph["w1"]=tf.constant(state["w1:0"], name="s_w1")
	g_graph["b1"]=tf.constant(state["b1:0"], name="s_b1")
	#g_graph["w1"]=tf.Variable(tf.truncated_normal(getShape( tg, "w1"), stddev=0.1), name="w1")
	#g_graph["b1"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b1")), name="b1")
	
	#             tf.nn.relu(tf.nn.conv2d(x_image,            w1,            strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + b1,            name="h1")
	g_graph["h1"]=tf.nn.relu(tf.nn.conv2d(g_graph["X"], g_graph["w1"], strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + g_graph["b1"], name="s_h1")
	# 2x2 max pooling
	g_graph["h1pooled"] = tf.nn.max_pool(g_graph["h1"], ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStride, 2, 1], padding='SAME', name="s_h1_pooled")

	g_graph["w2"]=tf.constant(state["w2:0"], name="s_w2")
	g_graph["b2"]=tf.constant(state["b2:0"], name="s_b2")
	#g_graph["w2"]=tf.Variable(tf.truncated_normal(getShape( tg, "w2"), stddev=0.1), name="w2")
	#g_graph["b2"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b2")), name="b2")

	g_graph["h2"]=tf.nn.relu(tf.nn.conv2d(g_graph["h1pooled"], g_graph["w2"], strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + g_graph["b2"], name="s_h2")

	g_graph["h2pooled"] = tf.nn.max_pool(g_graph["h2"], ksize=[1, k_poolRows, 2, 1], strides=[1, k_poolStride, 2, 1], padding='SAME', name='s_h2_pooled')
	g_graph["convlayers_output"] = tf.reshape(g_graph["h2pooled"], [-1, k_downsampledWidth * k_downsampledHeight*L2_CHANNELS]) # to prepare it for multiplication by W_fc1

#
	g_graph["W_fc1"] = tf.constant(state["W_fc1:0"], name="s_W_fc1")
	g_graph["b_fc1"] = tf.constant(state["b_fc1:0"], name="s_b_fc1")

	#g_graph["keepProb"]=tf.placeholder(tf.float32, (), name= "keepProb")
	#g_graph["h_fc1"] = tf.nn.relu(tf.matmul(tf.nn.dropout(g_graph["convlayers_output"], g_graph["keepProb"]), g_graph["W_fc1"]) + g_graph["b_fc1"], name="h_fc1")
	g_graph["h_fc1"] = tf.nn.relu(tf.matmul(g_graph["convlayers_output"], g_graph["W_fc1"]) + g_graph["b_fc1"], name="s_h_fc1")


	#Read out layer
	g_graph["W_fc2"] = tf.constant(state["W_fc2:0"], name="s_W_fc2")
	g_graph["b_fc2"] = tf.constant(state["b_fc2:0"], name="s_b_fc2")


	g_graph["logits_"] = tf.matmul(g_graph["h_fc1"], g_graph["W_fc2"])
	g_graph["logits"] = tf.add(g_graph["logits_"] , g_graph["b_fc2"] , name="s_logits")


	g_graph["softmax_preds"] = tf.nn.softmax(logits=g_graph["logits"], name="s_softmax_preds")


	return g_graph



def load(pickleFile, randomize=0) :
	print(' will read state from ' + pickleFile)
	state = pickle.load( open( pickleFile, "rb" ) )

	if randomize ==1 :
		print('randomizing weights')
		for n in state.keys():
			print('shape of state[' + n + '] is ' + str(state[n].shape))
			state[n] = 2* np.random.random_sample(state[n].shape).astype(np.float32) -1

	return constructSTModel(state)

