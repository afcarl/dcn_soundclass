import tensorflow as tf
import numpy as np

k_freqbins=256
k_width=856

def getTrainedModel(model_file) :

	st_saver = tf.train.import_meta_graph(model_file)
	with tf.Session() as sess:

		# Do i also have to restore to get the varaibale value?? 
		st_saver.restore(sess, tf.train.latest_checkpoint('logs.2017.04.18/mtl_2.or_channels.epsilon_1.0/checkpoints/'))
		print ('...and allvars :')  #probalby have to restore from checkpoint first
		all_vars = tf.get_collection('vars')
		for v in all_vars:
			v_ = sess.run(v)
			print(v_)

	# Access the graph
	st_graph = tf.get_default_graph()
	return st_graph


g = getTrainedModel('logs.2017.04.18/mtl_2.or_channels.epsilon_1.0/my-model.meta')

#vnamelist = [n.name for n in g.as_graph_def().node]
vnamelist =[n.name for n in tf.global_variables()]
print('----Variables in graph are : ' + str(vnamelist))

#opslist  = [n.name for n in g.get_operations()] 
#print('----Operatios in graph are : ' + str(opslist))

print ('...and allvars :')  #probalby have to restore from checkpoint first
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)

#
print(' here we go ........')

tf.GraphKeys.USEFUL = 'useful'
var_list = tf.get_collection(tf.GraphKeys.USEFUL)

####tf.add_to_collection(tf.GraphKeys.USEFUL, X)    #input place holder
####tf.add_to_collection(tf.GraphKeys.USEFUL, keepProb) #place holder
####tf.add_to_collection(tf.GraphKeys.USEFUL, softmax_preds)
####tf.add_to_collection(tf.GraphKeys.USEFUL, h1)
####tf.add_to_collection(tf.GraphKeys.USEFUL, h2)

#X = g.get_tensor_by_name('X/Adam:0')# placeholder for input
#X = tf.placeholder(tf.float32, [None,k_freqbins*k_width], name= "X")
X=var_list[0]
print('X is ' + str(X))

#keepProb = g.get_tensor_by_name('keepProb')
#keepProb=tf.placeholder(tf.float32, (), name= "keepProb")
keepProb=var_list[1]
print('keepProb is ' + str(keepProb))


softmax_preds=var_list[2]
assert softmax_preds.graph is tf.get_default_graph()
rimages=np.random.uniform(0.,1., (3,k_freqbins*k_width))


print('got my image, ready to run!')

#Z = tf.placeholder(tf.float32, [k_freqbins*k_width], name= "Z")
#Y=tf.Variable(tf.truncated_normal([k_freqbins*k_width], stddev=0.1), name="Y")
#Y=tf.assign(Y,Z)

#with tf.Session() as sess:
#	sess.run ( tf.global_variables_initializer ())
#	foo = sess.run(Y, feed_dict={Z: rimage})

with tf.Session() as sess:
	sess.run ( tf.global_variables_initializer ())
	predictions = sess.run(softmax_preds, feed_dict ={ X : rimages ,  keepProb : 1.0 })
	print('predictions are : ' + str(predictions))

