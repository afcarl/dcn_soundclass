"""

--------------------------------------------------------------------------
"""
import tensorflow as tf
import numpy as np
import spectreader
import os
import time

# get args from command line
import argparse
FLAGS = None
# ------------------------------------------------------
# get any args provided on the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--outdir', type=str, help='output directory for logging',  default='.') 
parser.add_argument('--numClasses', type=int, help='number of classes in data', choices=[2,50], default=2) #default for testing
parser.add_argument('--checkpointing', help='True/False - for both saving and starting from checkpoints', default=False)
parser.add_argument('--checkpointPeriod', type=int, help='checkpoint every n batches', default=8) 

parser.add_argument('--learning_rate', type=float, help='learning rate', default=.001) 
parser.add_argument('--batchsize', type=int, help='number of data records per training batch', default=8) #default for testing
parser.add_argument('--n_epochs', type=int, help='number of epochs to use for training', default=2) #default for testing
parser.add_argument('--keepProb', type=float, help='keep probablity for dropout before 1st fully connected layer during training', default=1.0) #default for testing

FLAGS, unparsed = parser.parse_known_args()
print('\n FLAGS parsed :  {0}'.format(FLAGS))

#HARD-CODED data-dependant parameters ------------------
#dimensions of image (pixels)
k_height=256
k_width=856

k_numClasses=FLAGS.numClasses  #hard coded, depends upon data
validationSamples=8*k_numClasses
trainingSamples=32*k_numClasses

# ------------------------------------------------------
# Define paramaters for the model
learning_rate = FLAGS.learning_rate
k_batchsize = FLAGS.batchsize 
n_epochs = FLAGS.n_epochs #6  #NOTE: we can load from checkpoint, but new run will last for n_epochs anyway

k_batchesPerLossReport= 4  #writes loss to the console every n batches

K_ConvRows=256
K_ConvCols=5
k_ConvStrideRows=1
k_ConvStrideCols=1

L1_CHANNELS=32
L2_CHANNELS=64
FC_SIZE = 32

k_keepProb=FLAGS.keepProb


# Derived parameters for convenience (do not change these)
k_vbatchsize = k_batchsize
k_numVBatches = validationSamples/k_vbatchsize
print(' ------- For validation, will run ' + str(k_numVBatches) + ' batches of ' + str(k_vbatchsize) + ' datasamples')
#k_BATCHESPEREPOCH = trainingSamples/k_batchsize
#k_TOTALBATCHES = n_epochs*k_BATCHESPEREPOCH

# ------------------------------------------------------
#Other non-data, non-model params
CHECKPOINTING=FLAGS.checkpointing
k_checkpointPeriod = 8  # in units of batches

OUTDIR = FLAGS.outdir

CHKPOINTDIR = OUTDIR + '/checkpoints' # create folder manually
CHKPTBASE =  CHKPOINTDIR + '/model.ckpt'	# base name used for checkpoints
LOGDIR = OUTDIR + '/log_graph'			#create folder manually
#OUTPUTDIR = i_outdir

NUM_THREADS = 4  #used for enqueueing TFRecord data 
#=============================================

def getImage(fnames, nepochs=None) :
    """ Reads data from the prepaired *list* files in fnames of TFRecords, does some preprocessing 
    params:
    fnames - list of filenames to read data from
    nepochs - An integer (optional). Just fed to tf.string_input_producer().  Reads through all data num_epochs times before generating an OutOfRange error. None means read forever.
    """
    label, image = spectreader.getImage(fnames, nepochs)
    image=tf.reshape(image,[k_height*k_width])
    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, k_numClasses))
    return label, image

def get_datafiles(a_dir, startswith):
    """ Returns a list of files in a_dir that start with the string startswith.
    e.g. e.g. get_datafiles('data', 'train-') 
    """ 
    return  [a_dir + '/' + name for name in os.listdir(a_dir)
            if name.startswith(startswith)]

#=============================================
# Step 1: Read in data

# getImage reads data for enqueueing shufflebatch, shufflebatch manages it's own dequeing 
# ---- First set up the graph for the TRAINING DATA
target, data = getImage(get_datafiles('data'+ str(k_numClasses), 'train-'), n_epochs)

imageBatch, labelBatch = tf.train.shuffle_batch(
    [data, target], batch_size=k_batchsize,
    num_threads=NUM_THREADS,
    allow_smaller_final_batch=True, #want to finish an eposh even if datasize doesn't divide by batchsize
    enqueue_many=False, #IMPORTANT to get right, default=False - 
    capacity=1000,  #1000,
    min_after_dequeue=500) #500

# ---- same for the VALIDATION DATA
vtarget, vdata = getImage(get_datafiles('data'+ str(k_numClasses), 'validation-')) # one "epoch" for validation

#vimageBatch, vlabelBatch = tf.train.shuffle_batch(
#    [vdata, vtarget], batch_size=k_vbatchsize,
#    num_threads=NUM_THREADS,
#    allow_smaller_final_batch=True, #want to finish an eposh even if datasize doesn't divide by batchsize
#    enqueue_many=False, #IMPORTANT to get right, default=False - 
#    capacity=1000,  #1000,
#    min_after_dequeue=500) #500

vimageBatch, vlabelBatch = tf.train.batch(
    [vdata, vtarget], batch_size=k_vbatchsize,
    num_threads=NUM_THREADS,
    allow_smaller_final_batch=False, #want to finish an eposh even if datasize doesn't divide by batchsize
    enqueue_many=False, #IMPORTANT to get right, default=False - 
    capacity=1000)

# Step 2: create placeholders for features (X) and labels (Y)
# each lable is one hot vector.
# 'None' here allows us to fill the placeholders with different size batches (which we do with training and validation batches)
X = tf.placeholder(tf.float32, [None,k_height*k_width], name= "X")
x_image = tf.reshape(X, [-1,1,k_width,k_height])  # reshape so we can run a 2d convolutional net
Y = tf.placeholder(tf.float32, [None,k_numClasses], name= "Y")  #labeled classes, one-hot

# Step 3: create weights and bias

#Layer 1
# 1 input channel, L1_CHANNELS output channels
w1=tf.Variable(tf.truncated_normal([1, K_ConvCols, K_ConvRows, L1_CHANNELS], stddev=0.1), name="w1")
b1=tf.Variable(tf.constant(0.1, shape=[L1_CHANNELS]), name="b1")

h1=tf.nn.relu(tf.nn.conv2d(x_image, w1, strides=[1, 1, k_ConvStrideCols, k_ConvStrideRows], padding='SAME') + b1)
# 2x2 max pooling
h1pooled = tf.nn.max_pool(h1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

#Layer 2
#L1_CHANNELS input channels, L2_CHANNELS output channels
w2=tf.Variable(tf.truncated_normal([1, K_ConvCols, L1_CHANNELS, L2_CHANNELS], stddev=0.1), name="w2")
b2=tf.Variable(tf.constant(0.1, shape=[L2_CHANNELS]), name="b2")

h2=tf.nn.relu(tf.nn.conv2d(h1pooled, w2, strides=[1, 1, k_ConvStrideCols, 1], padding='SAME') + b2)

with tf.name_scope ( "Conv_layers_out" ):
	h2pooled = tf.nn.max_pool(h2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME', name='h2_pooled')
	h_pool2_flat = tf.reshape(h2pooled, [-1, (k_width/4) * 1*L2_CHANNELS]) # to prepare it for multiplication by W_fc1

#h2pooled is number of pixels / 2 / 2  (halved in size at each layer due to pooling)
# check our dimensions are a multiple of 4
if (k_width%4 ):
	print ('Error: width and height must be a multiple of 4')
	sys.exit(1)

#now do a fully connected layer: every output connected to every input pixel of each channel
W_fc1 = tf.Variable(tf.truncated_normal([(k_width/4) * 1 * L2_CHANNELS, FC_SIZE], stddev=0.1), name="W_fc1")
b_fc1 = tf.Variable(tf.constant(0.1, shape=[FC_SIZE]) , name="b_fc1")

keepProb=tf.placeholder(tf.float32, (), name= "keepProb")
h_fc1 = tf.nn.relu(tf.matmul(tf.nn.dropout(h_pool2_flat, keepProb), W_fc1) + b_fc1)

#Read out layer
W_fc2 = tf.Variable(tf.truncated_normal([FC_SIZE, k_numClasses], stddev=0.1), name="W_fc2")
b_fc2 = tf.Variable(tf.constant(0.1, shape=[k_numClasses]), name="b_fc2")


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
#could do a dropout here on h
logits = tf.matmul(h_fc1, W_fc2) + b_fc2


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
summaryloss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
meanloss = tf.reduce_mean(summaryloss)

# Step 6: define training op
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(meanloss, global_step=global_step)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(meanloss, global_step=global_step)
# NOTE: Must save global step here if you are doing checkpointing and expect to start from step where you left off.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(meanloss, global_step=global_step)


#---------------------------------------------------------------
# VALIDATE
#--------------------------------------------------------------
# The nodes are used for running the validation data and getting accuracy scores from the logits
with tf.name_scope("VALIDATION"):
	preds = tf.nn.softmax(logits=logits, name="validation_softmax")
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	batchNumCorrect = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(

	# All this, just to feed a friggin float computed over several batches into a tensor we want to use for a summary
	validationtensor = tf.Variable(0.0, trainable=False, name="validationtensor")
	wtf = tf.placeholder(tf.float32, ())
	summary_validation = tf.assign(validationtensor, wtf)

# Run the validation set through the model and compute statistics to report as summaries
def validate(sess, printout=False) : 
	with tf.name_scope ( "summaries" ):
		# test the model
		total_correct_preds = 0

		try:
			for i in range(k_numVBatches):
				
				X_batch, Y_batch = sess.run([vimageBatch, vlabelBatch])
				batch_correct, predictions = sess.run([batchNumCorrect, preds], feed_dict ={ X : X_batch , Y : Y_batch, keepProb : 1.}) 
				
				total_correct_preds +=  batch_correct
				#print (' >>>>  Batch " + str(i) + ' with batch_correct = ' + str(batch_correct) + ', and total_correct is ' + str(total_correct_preds))

				if printout:
					print(' labels for batch:')
					print(Y_batch)
					print(' predictions for batch')
					print(predictions)
					# print num correct for each batch
					print(u'(Validation batch) num correct for batchsize of {0} is {1}'.format(k_vbatchsize , batch_correct))


			print (u'(Validation EPOCH) num correct for EPOCH size of {0} ({1} batches) is {2}'.format(validationSamples , i+1 , total_correct_preds))
			print('so the percent correction for validation set = ' + str(total_correct_preds/validationSamples))
			msummary = sess.run(mergedvalidation, feed_dict ={ X : X_batch , Y : Y_batch, wtf : total_correct_preds/validationSamples, keepProb : 1.}) #using last batch to computer loss for summary
			

		except Exception, e:
			print e

		return msummary


#--------------------------------------------------------------
#   Visualize with Tensorboard
# -------------------------------------------------------------

def create_train_summaries ():
		with tf.name_scope ( "train_summaries" ):
			tf.summary.scalar ( "mean_loss" , meanloss)
			return tf.summary.merge_all ()

mergedtrain = create_train_summaries()

def create_validation_summaries ():
		with tf.name_scope ( "validation_summaries" ):
			#tf.summary.scalar ( "validation_correct" , batchNumCorrect)
			tf.summary.scalar ( "summary_validation", summary_validation)
			return tf.summary.merge_all ()

mergedvalidation = create_validation_summaries()

# --------------------------------------------------------------
# TRAIN
#---------------------------------------------------------------
def trainModel():

	with tf.Session() as sess:
		writer = tf.summary.FileWriter(LOGDIR)  # for logging
		saver = tf.train.Saver() # for checkpointing

		#### Must run local initializer if nepochs arg to getImage is other than None!
		#sess.run(tf.local_variables_initializer())
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

		#not doing it here, but global_step could have been initialized by a checkpoint
		if CHECKPOINTING :
			ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHKPTBASE))
		else :
			ckpt = False
		if ckpt and ckpt.model_checkpoint_path:
			print('Checkpointing restoring from path ' + ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			#only save graph if we are not starting run from a checkpoint
			writer.add_graph(sess.graph)

 
		initial_step = global_step.eval()
		print('initial step will be ' + str(initial_step)) # non-zero if check pointing
		batchcount=initial_step
		start_time = time.time()
		
		# Create a coordinator, launch the queue runner threads.
		coord = tf.train.Coordinator()
		enqueue_threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		
		try:
			batchcountloss = 0 #for reporting purposes
			while True: # for each batch, until data runs out
				if coord.should_stop():
					break

				X_batch, Y_batch = sess.run([imageBatch, labelBatch])
				_, loss_batch = sess.run([optimizer, meanloss], feed_dict ={ X : X_batch , Y : Y_batch, keepProb : k_keepProb })   #DO WE NEED meanloss HERE? Doesn't optimer depend on it?
				batchcountloss += loss_batch

				batchcount += 1
				if (not batchcount%k_batchesPerLossReport) :
					print('batchcount = ' + str(batchcount))
					avgBatchLoss=batchcountloss/k_batchesPerLossReport
					print(u'Average loss per batch {0}: {1}'.format(batchcount, avgBatchLoss))
					batchcountloss=0

					tsummary = sess.run(mergedtrain, feed_dict ={ X : X_batch , Y : Y_batch, keepProb : 1.0 }) #?? keep prob ??
					writer.add_summary(tsummary, global_step=batchcount)

					vsummary=validate(sess)
					writer.add_summary(vsummary, global_step=batchcount)


				if not (batchcount  % k_checkpointPeriod) :
					saver.save(sess, CHKPTBASE, global_step=batchcount)

		except tf.errors.OutOfRangeError, e:  #done with training epochs. Validate once more before closing threads
			# So how, finally?
			print('ok, let validate ------------------------------')

			vsummary=validate(sess)
			writer.add_summary(vsummary, global_step=batchcount+1)


			coord.request_stop(e)

		except Exception, e:	
			print('train: WTF')
			print e

		finally :
			coord.request_stop()
			coord.join(enqueue_threads)
			writer.close()
		
		# grab the total training time
		totalruntime = time.time() - start_time
		print 'Total training time: {0} seconds'.format(totalruntime)
		print(' Finished!') # should be around 0.35 after 25 epochs
		print(' ===============================================================') 

#=============================================================================================
# Do it
trainModel()
