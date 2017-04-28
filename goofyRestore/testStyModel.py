"""
eg 
python testModel.py  logs.2017.04.21/mtl_2.or_channels.epsilon_1.0/my-model.meta  logs.2017.04.21/mtl_2.or_channels.epsilon_1.0/checkpoints/

"""
import tensorflow as tf
import numpy as np
import styModel

from PIL import TiffImagePlugin
from PIL import Image

# get args from command line
import argparse
FLAGS = None

k_height=1
k_width=856
k_inputChannnels =256

VERBOSE=False
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('metamodel', type=str, help='stored graph'  ) 
parser.add_argument('checkptDir', type=str, help='the checkpoint directory from where the latest checkpoint will be read to restore values for variables in the graph'  ) 
FLAGS, unparsed = parser.parse_known_args()

k_freqbins=256
k_width=856

styg = styModel.load(FLAGS.metamodel, FLAGS.checkptDir)
#print('got my graph! ' + str(g))
if VERBOSE : 
	vnamelist =[n.name for n in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
	print('TRAINABLE vars:')
	for n in vnamelist :
		print(n)


print(' here we go ........')


def soundfileBatch(slist) :
	#looks weird, and it might be possible to simplify, but this is what it takes to get it into the right shape.
	# This is the shape that the training network passes to the optimizer after it's load and reshaping of an image.
	#return [np.reshape(np.transpose(np.array(Image.open(name).point(lambda i: i*255)).flatten()), [1,k_height,k_width,k_inputChannnels]) for name in slist ]
	return( [np.transpose(np.reshape(np.array(Image.open(name).point(lambda i: i*255)), [1,k_freqbins,k_width,1]), [0,3,2,1]) for name in slist ])

#just test the validation set 
#Flipping and scaling seem to have almost no effect on the clasification accuracy
rimages=soundfileBatch(['data2/validate/205 - Chirping birds/5-242490-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-242491-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-243448-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-243449-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-243450-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-243459-A._2_.tif',
	'data2/validate/205 - Chirping birds/5-243459-B._2_.tif',
	'data2/validate/205 - Chirping birds/5-257839-A._2_.tif',
	'data2/validate/101 - Dog/5-203128-A._1_.tif',
	'data2/validate/101 - Dog/5-203128-B._1_.tif',
	'data2/validate/101 - Dog/5-208030-A._1_.tif',
	'data2/validate/101 - Dog/5-212454-A._1_.tif',
	'data2/validate/101 - Dog/5-213855-A._1_.tif',
	'data2/validate/101 - Dog/5-217158-A._1_.tif',
	'data2/validate/101 - Dog/5-231762-A._1_.tif',
	'data2/validate/101 - Dog/5-9032-A._1_.tif',
	])

im=np.empty([1,1,k_width,k_freqbins ])

with tf.Session() as sess:

	predictions=[]
#	#sess.run ( tf.global_variables_initializer ())
#	#savior.restore(sess, tf.train.latest_checkpoint(FLAGS.checkptDir))
	styModel.initialize_variables(sess)
	#print('ok, all initialized')
	if 0 :
		print ('...GLOBAL_VARIABLES :')  #probalby have to restore from checkpoint first
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)

	if 0 :
		for v in ["s_w1:0", "s_b1:0", "s_w2:0", "s_b2:0", "s_W_fc1:0", "s_b_fc1:0", "s_W_fc2:0", "s_b_fc2:0"] :
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v)))


	if 1 :
		for v in ["s_h1:0"] :
			#im = np.reshape(np.transpose(rimages[6]), [1,k_width*k_freqbins ])
			im=rimages[6]
			sess.run(styg["X"].assign(im)) #transpose to make freqbins channels
			print(tf.get_default_graph().get_tensor_by_name(v))
			print(sess.run(tf.get_default_graph().get_tensor_by_name(v)))



	print('predictions are : ')
	for im_ in rimages :
		#im = np.reshape(np.transpose(im_), [1,k_width*k_freqbins ])
		im=im_
		sess.run(styg["X"].assign(im)) #transpose to make freqbins channels
		prediction = sess.run(styg["softmax_preds"])
		print(str(prediction[0]))
		#predictions.extend(prediction[0])



