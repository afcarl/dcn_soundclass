
""" An implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function
import sys 

import os
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils

if len(sys.argv) != 5 :
    print('Usage python style_transfer.py content.noext style.noext noise outputDirectory')
    sys.exit()
else :
    i_content = sys.argv[1]
    i_style = sys.argv[2]
    i_noise=float(sys.argv[3])
    i_outdir=sys.argv[4]
    print ('Argument List:', i_content, i_style, i_noise)

CHECKPOINTING=True

FILETYPE = ".tif"
# parameters to manage experiments
STYLE = i_style
CONTENT = i_content
STYLE_IMAGE = 'styles/' + STYLE + FILETYPE
CONTENT_IMAGE = 'content/' + CONTENT + FILETYPE
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 856
  # This seems to be the paramter that really controls the balance between content and style
  # The more noise, the less content
NOISE_RATIO = i_noise # percentage of weight of the noise for intermixing with the content image

# Layers used for style features. You can change this.
STYLE_LAYERS = ['h1', 'h2']
W = [0.5, 1.0, 1.5, 3.0, 4.0] # give more weights to deeper layers.

# Layer used for content features. You can change this.
CONTENT_LAYER = 'h2'

#Relationship a/b is 1/20
ALPHA = 10  
BETA = 200

LOGDIR = './log_graph'			#create folder manually
CHKPTDIR =  './checkpoints'		# create folder manually
OUTPUTDIR = i_outdir

ITERS = 1
LR = 2.0

#MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
MEAN_PIXELS = np.array([128, 128, 128]).reshape((1,1,1,3))
""" MEAN_PIXELS is defined according to description on their github:
https://gist.github.com/ksimonyan/211839e770f7b538e2d8
'In the paper, the model is denoted as the configuration D trained with scale jittering. 
The input images should be zero-centered by mean pixel (rather than mean image) subtraction. 
Namely, the following BGR values should be subtracted: [103.939, 116.779, 123.68].'
"""

# VGG-19 parameters file
#VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
#VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
#EXPECTED_BYTES = 534904783

def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.
    
    Inputs: 
        p, f are just P, F in the paper 
        (read the assignment handout if you're confused)
        Note: we won't use the coefficient 0.5 as defined in the paper
        but the coefficient as defined in the assignment handout.
    Output:
        the content loss

    """
    pdims=p.shape
    #print('p has dims : ' + str(pdims)) 
    coef = np.multiply.reduce(pdims)   # Hmmmm... maybe don't want to include the first dimension
    #this makes the loss 0!!!
    #return (1/4*coef)*tf.reduce_sum(tf.square(f-p))
    return tf.reduce_sum((f-p)**2)/(4*coef)


def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Hint: you'll first have to reshape F

        inputs: F: the tensor of all feature channels in a given layer
                N: number of features (channels) in the layer
                M: the total number of filters in each filter (length * height)

        F comes in as numchannels*length*height, and 
    """
        # We want to reshape F to be number of feaures (N) by the values in the feature array ( now represented in one long vector of length M) 

    Fshaped = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(Fshaped), Fshaped) # return G of size #channels x #channels


def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer
    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image
    Output:
        the style loss at a certain layer (which is E_l in the paper)

    Hint: 1. you'll have to use the function _gram_matrix()
        2. we'll use the same coefficient for style loss as in the paper
        3. a and g are feature representation, not gram matrices
    """
    horizdim = 1  # recall that first dimension of tensor is minibatch size
    vertdim = 2
    featuredim = 3



    # N - number of features
    N = a.shape[featuredim]  #a & g are the same shape
    # M - product of first two dimensions of feature map
    M = a.shape[horizdim]*a.shape[vertdim]

    #print(' N is ' + str(N)  + ', and M is ' + str(M))
    
    # This is 'E' from the paper and the homework handout.
    # It is a scalar for a single layer
    diff = _gram_matrix(a, N, M)-_gram_matrix(g, N, M)
    sq = tf.square(diff)
    s=tf.reduce_sum(sq)
    return (s/(4*N*N*M*M))
    

def _create_style_loss(A, model):
    """ Return the total style loss
    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]
    
    #print('E is ' + str(tf.shape(E)))
    #print('W is ' + str(tf.shape(W)))
    ###############################
    ## TO DO: return total style loss
    foo =  (sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))]))
    #print('foo  is ' + str(foo))
	#    return (sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))]))
    return np.dot(W, E)
    ###############################

def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image)) # assign content image to the input variable
            # model[CONTENT_LAYER] is a relu op
            p = sess.run(model[CONTENT_LAYER])
        # ??????????????????????????????????????????????????????
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])                              
        style_loss = _create_style_loss(A, model)

        ##########################################
        ## TO DO: create total loss. 
        ## Hint: don't forget the content loss and style loss weights
        total_loss = ALPHA*content_loss + BETA*style_loss
        ##########################################

    return content_loss, style_loss, total_loss

def _create_summary(model):
    """ Create summary ops necessary
        Hint: don't forget to merge them
    """
    with tf.name_scope ( "summaries" ):
        tf.summary.scalar ( "content loss" , model['content_loss'])
        tf.summary.scalar ( "style_loss" , model['style_loss'])
        tf.summary.scalar ( "total_loss" , model['total_loss'])
        # because you have several summaries, we should merge them all
        # into one op to make it easier to manage
        return tf.summary.merge_all()


def train(model, generated_image, initial_image):
    """ Train your model.
    Don't forget to create folders for checkpoints and outputs.
    """
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ###############################
        ## TO DO: 
        ## 1. initialize your variables
        ## 2. create writer to write your graph
        sess.run ( tf.global_variables_initializer ())
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        ###############################
        sess.run(input_image.assign(initial_image))
        if CHECKPOINTING :
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHKPTDIR + '/checkpoint'))
        else :
            ckpt = False

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()
        
        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20
            
            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:
                ###############################
                ## TO DO: obtain generated image and loss
                # following the optimazaiton step, calculate loss

                 # get the modified image
                gen_image = sess.run(input_image)
                
                model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                input_image, content_image, style_image)
                summary = sess.run(model['summary_op'])
               
               	# reassign the input to be the generated image from the last iteration
                sess.run(input_image.assign(gen_image))
                ###############################
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(sess.run(model['total_loss']))) #???????
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = OUTPUTDIR + '/%d.tif' % (index)
                utils.save_image(filename, gen_image[0])

                if (index + 1) % 20 == 0:
                    saver.save(sess, CHKPTDIR + '/style_transfer', index)

        writer.close()

#took this out of main to make input_image GLOBAL
#def main():

print('RUN MAIN')
with tf.variable_scope('input') as scope:
	# use variable instead of placeholder because we're training the intial image to make it
	# look like both the content image and the style image
	input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)


utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
model = vgg_model.load_vgg(VGG_MODEL, input_image)
model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
content_image = content_image - MEAN_PIXELS
style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
style_image = style_image - MEAN_PIXELS

model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model, 
                                                input_image, content_image, style_image)
###############################
## TO DO: create optimizer
## model['optimizer'] = ...
model['optimizer'] =  tf.train.AdamOptimizer(LR).minimize(model['total_loss'])
###############################
model['summary_op'] = _create_summary(model)

initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
#def train(model, generated_image, initial_image):
train(model, input_image, initial_image)

#if __name__ == '__main__':
#    main()
