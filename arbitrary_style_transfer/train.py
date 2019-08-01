# Train the Style Transfer Net

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from style_transfer_net import StyleTransferNet
from utils import get_train_images

tmp_save_path = r'tmp_model'
STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)

EPOCHS = 4
EPSILON = 1e-5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

CONTENT_LOSS = []
WEIGHTED_STYLE_LOSS = []
LOSS = []
STEP = []
LR = []

def train(style_weight, content_imgs_path, style_imgs_path, encoder_path, 
          model_save_path, debug=False, logging_period=100):
    print("epoch = ", EPOCHS)
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # guarantee the size of content and style images to be a multiple of BATCH_SIZE
    num_imgs = min(len(content_imgs_path), len(style_imgs_path))
    len_style_image = len(style_imgs_path)
    len_content_img = len(content_imgs_path)
    content_imgs_path = content_imgs_path[(len_content_img-len_style_image-1):-1]
    style_imgs_path   = style_imgs_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        content_imgs_path = content_imgs_path[:-mod]
        style_imgs_path   = style_imgs_path[:-mod]
    print(len(content_imgs_path))
    print(len(style_imgs_path))
    print("the last content_img_path is", content_imgs_path[-1])
    print("the last style_img_path   is", style_imgs_path[-1])
    print("the number of img is " + str(num_imgs - mod))
    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        style   = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='style')

        # create the style transfer net
        stn = StyleTransferNet(encoder_path)

        # pass content and style to the stn, getting the generated_img
        generated_img = stn.transform(content, style)

        # get the target feature maps which is the output of AdaIN
        target_features = stn.target_features

        # pass the generated_img to the encoder, and use the output compute loss
        generated_img = tf.reverse(generated_img, axis=[-1])  # switch RGB to BGR
        generated_img = stn.encoder.preprocess(generated_img) # preprocess image
        enc_gen, enc_gen_layers = stn.encoder.encode(generated_img)

        # compute the content loss
        # content_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(enc_gen - target_features), axis=[1, 2]))) #it is good
        content_loss = tf.sqrt(tf.reduce_sum(tf.reduce_mean(tf.square(enc_gen - target_features), axis=[1, 2]))) #just to compare

        # compute the style loss
        style_layer_loss = []
        for layer in STYLE_LAYERS:
            enc_style_feat = stn.encoded_style_layers[layer]
            enc_gen_feat   = enc_gen_layers[layer]

            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat,   [1, 2])

            sigmaS = tf.sqrt(varS + EPSILON)
            sigmaG = tf.sqrt(varG + EPSILON)

            #l2_mean  = tf.sqrt(tf.reduce_sum(tf.square(meanG - meanS)))  #it is good
            #l2_sigma = tf.sqrt(tf.reduce_sum(tf.square(sigmaG - sigmaS))) #it is good

            l2_mean = tf.reduce_sum(tf.square(meanG - meanS)) #just to compare
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))#just to compare

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # compute the total loss
        loss = content_loss + style_weight * style_loss

        # Training step
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        # saver
        saver = tf.train.Saver(max_to_keep=10)

        ###### Start Training ######
        step = 0
        n_batches = int(len(content_imgs_path) // BATCH_SIZE)

        if debug:
            elapsed_time = datetime.now() - start_time
            start_time = datetime.now()
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')

        try:
            for epoch in range(EPOCHS):

                np.random.shuffle(content_imgs_path)
                np.random.shuffle(style_imgs_path)

                for batch in range(n_batches):
                    # retrive a batch of content and style images
                    content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                    style_batch_path   = style_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                    content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                    style_batch   = get_train_images(style_batch_path,   crop_height=HEIGHT, crop_width=WIDTH)

                    # run the training step
                    sess.run(train_op, feed_dict={content: content_batch, style: style_batch})

                    step += 1

                    if step % 1000 == 0:
                        saver.save(sess, model_save_path, global_step=step, write_meta_graph=False)

                    if debug:
                        is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                        if is_last_step or step == 1 or step % logging_period == 0:
                            elapsed_time = datetime.now() - start_time
                            _content_loss, _style_loss, _loss = sess.run([content_loss, style_loss, loss], 
                                feed_dict={content: content_batch, style: style_batch})
                            CONTENT_LOSS.append(_content_loss)
                            WEIGHTED_STYLE_LOSS.append(style_weight * _style_loss)
                            LOSS.append(_loss)
                            STEP.append(step)
                            print('step: %d,  total loss: %.3f,  elapsed time: %s' % (step, _loss, elapsed_time))
                            print('content loss: %.3f' % (_content_loss))
                            print('style loss  : %.3f,  weighted style loss: %.3f\n' % (_style_loss, style_weight * _style_loss))
        except Exception as ex:
            saver.save(sess, model_save_path, global_step=step)
            print('\nSomething wrong happens! Current model is saved to <%s>' % tmp_save_path)
            print('Error message: %s' % str(ex))

        ###### Done Training & Save the model ######
        saver.save(sess, model_save_path)

        if debug:
            elapsed_time = datetime.now() - start_time
            print('Done training! Elapsed time: %s' % elapsed_time)
            print('Model is saved to: %s' % model_save_path)

        loss_file = r'loss_trainging/loss.txt'
        content_loss_file = r'loss_trainging/content_loss.txt'
        style_loss_file = r'loss_trainging/style_loss.txt'
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.plot(STEP, LOSS, 'r', label='LOSS')
        plt.plot(STEP, CONTENT_LOSS, 'b', label='CONTENT_LOSS')
        plt.plot(STEP, WEIGHTED_STYLE_LOSS, 'g', label='WEIGHTED_STYLE_LOSS')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
                   ncol=4, mode="expand", borderaxespad=0.)
        plt.savefig(r"models/comic_sqrt_0.1_face(change_loss_to_the_origin_sqrt)/comic_0.1_face_sqrt.jpg")

        with open(loss_file, 'w') as f:
            f.write(LOSS)
        with open(content_loss_file, 'w') as f:
            f.write(CONTENT_LOSS)
        with open(style_loss_file, 'w') as f:
            f.write(WEIGHTED_STYLE_LOSS)
        plt.show()
