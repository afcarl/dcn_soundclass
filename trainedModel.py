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

k_freqbins=256
k_width=856

k_verbose=0

def load(meta_model_file, restore_chkptDir) :

	st_saver = tf.train.import_meta_graph(meta_model_file)
	# Access the graph
	st_graph = tf.get_default_graph()

	with tf.Session() as sess:
		# Do i also have to restore to get the varaibale value?? 
		st_saver.restore(sess, tf.train.latest_checkpoint(restore_chkptDir))
		if k_verbose :
			print ('...GLOBAL_VARIABLES :')  #probalby have to restore from checkpoint first
			all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			for v in all_vars:
				v_ = sess.run(v)
				print(v_)
			print ('...WEIGHTS :')  #probalby have to restore from checkpoint first
			all_vars = tf.get_collection(tf.GraphKeys.WEIGHTS)
			for v in all_vars:
				v_ = sess.run(v)
				print(v_)

	return st_graph, st_saver

