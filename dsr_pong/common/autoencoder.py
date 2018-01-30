import tensorflow as tf
import numpy as np

class Autoencoder():

    def __init__(self,name,gridSize = 28,
                 input_channels = 1,
                 encoder_filter_channels = [32,64,64],
                 encoder_filter_sizes = [3,3,3],
                 encoder_strides = [2,2,2],
                 hidden_layers = [512],
                 embedding_size = 256,
                 decoder_filter_channels = [512,256,128,64],
                 decoder_filter_sizes = [4,4,4,4],
                 decoder_strides = [1,2,2,2],
                 decoder_padding = [0,0,1,0],
                 LOG_DIMENSIONS = True):
        with tf.name_scope(name):

            #Input for the autoencoder
            self.X = tf.placeholder(tf.uint8, shape=(None, gridSize, gridSize, input_channels), name= "Input")

            X_float = tf.cast(self.X,tf.float32) / 255.0

            batch_size = tf.shape(self.X)[0]
            print("Batch: %s" % batch_size)

            # Build the encoder block
            current_input = X_float
            self.encoder_weights = []
            self.encoder_biases = []
            shape = []

            for i in range(len(encoder_filter_sizes)):

                layername = "conv_enc_" + str(i)
                with tf.name_scope(layername):
                    shape.append(current_input.get_shape())

                    # Log the current shape
                    if LOG_DIMENSIONS:
                        print(shape[i])

                    n_input = shape[i].as_list()[3]
                    W = tf.Variable(tf.random_uniform(
                        shape=[encoder_filter_sizes[i], encoder_filter_sizes[i], n_input, encoder_filter_channels[i]],
                        minval=-1.0 / np.sqrt(n_input), maxval=1.0 / np.sqrt(n_input),
                        dtype=tf.float32), name= "W")
                    self.encoder_weights.append(W)
                    conv_layer = tf.nn.conv2d(current_input,
                                              filter=W,
                                              strides=[1, encoder_strides[i], encoder_strides[i], 1], padding='SAME')
                    b = tf.Variable(np.zeros(encoder_filter_channels[i]), dtype=tf.float32, name="b")
                    current_input = tf.nn.relu(tf.add(conv_layer, b))
                    self.encoder_biases.append(b)

            # Reshape the output of the convolutional layers
            conv_out_shape = current_input.get_shape().as_list()
            print("Conv_out: %s" % conv_out_shape)
            current_input = tf.reshape(current_input, shape=[batch_size, np.product(conv_out_shape[1:])])

            print("Hidden_in: %s" % current_input.get_shape())

            self.hidden_weights = []
            self.hidden_biases = []

            #Fully connected layer
            for hidden in hidden_layers:
                layername = "hidden_" + str(hidden)
                in_dim = current_input.get_shape().as_list()[1]
                with tf.name_scope(layername):
                    W = tf.Variable(tf.random_uniform(
                        shape=[in_dim,hidden],
                        minval=-1.0 / np.sqrt(in_dim), maxval=1.0 / np.sqrt(in_dim),
                        dtype=tf.float32), name= "W")
                    b = tf.Variable(np.zeros(hidden),name= "b",dtype= tf.float32)
                    current_input = tf.nn.relu(tf.add(tf.matmul(current_input,W),b))

                #keep track of the weighst
                self.hidden_weights.append(W)
                self.hidden_biases.append(b)

                print("Hidden: %s" % current_input.get_shape())

            with tf.name_scope("hidden_last"):
                #Transform to feature representation
                W_sr = tf.Variable(tf.random_uniform(
                        shape=[hidden_layers[-1], embedding_size],
                        minval=-1.0 / np.sqrt(hidden_layers[-1]), maxval=1.0 / np.sqrt(hidden_layers[-1]),
                        dtype=tf.float32), name= "W_sr")
                b_sr = tf.Variable(np.zeros(embedding_size),name= "b_sr",dtype= tf.float32)
                current_input =tf.add(tf.matmul(current_input,W_sr),b_sr)

            self.hidden_weights.append(W_sr)
            self.hidden_biases.append(b_sr)

            # compressed representation
            with tf.name_scope("Embedding"):
                self.z = tf.nn.tanh(current_input)

            if LOG_DIMENSIONS:
                print("z: %s" % str(self.z.get_shape()))

            # Build the decoder block
            #keep track of the decoder weights
            self.decoder_weights = []
            self.decoder_biases = []

            #first reshape the embeddings
            current_input = tf.reshape(self.z,[batch_size,1,1,embedding_size])

            print("Decoder_input: %s" % current_input.get_shape().as_list())

            current_input = self.create_conv_decoder(batch_size = batch_size,
                                     decoder_filter_channels = decoder_filter_channels,
                                     decoder_filter_sizes = decoder_filter_sizes,
                                     decoder_strides = decoder_strides,
                                     decoder_padding = decoder_padding,
                                     current_input= current_input)

            with tf.name_scope("Reconstruction"):
                self.Y = current_input

            with tf.name_scope("Loss"):
                self.cost = tf.reduce_mean(tf.square(X_float - self.Y))
                tf.summary.scalar("cost", self.cost)

    def create_conv_decoder(self,batch_size,decoder_filter_sizes,decoder_filter_channels,decoder_strides,decoder_padding,current_input):

        for i in range(len(decoder_filter_sizes)):
            layername = "conv_dec_" + str(i)
            with tf.name_scope(layername):
                out_size = decoder_strides[i] * (current_input.get_shape().as_list()[2] - 1) + decoder_filter_sizes[i] - 2 * decoder_padding[i]
                print(out_size)
                out_shape = [batch_size,out_size,out_size,decoder_filter_channels[i]]

                #compute the input dimension
                in_dim = current_input.get_shape().as_list()[3]

                W = tf.Variable(tf.random_uniform(
                    shape=[decoder_filter_sizes[i], decoder_filter_sizes[i],decoder_filter_channels[i],in_dim],
                    minval=-1.0 / np.sqrt(in_dim), maxval=1.0 / np.sqrt(in_dim),
                    dtype=tf.float32), name="W")

                if decoder_padding[i] == 0:
                    padding = 'VALID'
                else:
                    padding = 'SAME'

                conv_layer = tf.nn.conv2d_transpose(current_input,
                                                    filter=W,
                                                    output_shape= out_shape,
                                                    strides=[1, decoder_strides[i], decoder_strides[i], 1],padding= padding)
                b = tf.Variable(np.zeros(decoder_filter_channels[i]),dtype= tf.float32, name="b")
                current_input = tf.nn.relu(tf.add(conv_layer,b))
                current_input = tf.reshape(current_input,out_shape)

                #keep track of the variables
                self.decoder_weights.append(W)
                self.decoder_biases.append(b)

                print("Decoder: %s" % current_input.get_shape())

        with tf.name_scope("conv_dec_last"):
            #create the last layer
            in_dim = current_input.get_shape().as_list()[3]
            out_shape = self.X.get_shape().as_list()
            W = tf.Variable(tf.random_uniform(
                        shape=[4, 4,out_shape[3],in_dim],
                        minval=-1.0 / np.sqrt(in_dim), maxval=1.0 / np.sqrt(in_dim),
                        dtype=tf.float32), name="W")
            b = tf.Variable(np.zeros(out_shape[3]), dtype=tf.float32, name="b")
            conv_layer = tf.nn.conv2d_transpose(current_input,
                                                filter= W,
                                                output_shape= [batch_size,out_shape[1],out_shape[2],out_shape[3]],
                                                strides= [1,2,2,1])
            current_input = tf.nn.relu(tf.add(conv_layer,b))

        #keep track of the weights
        self.decoder_weights.append(W)
        self.decoder_biases.append(b)
        return current_input

    def copy_weights(self,other_enc):
        with tf.name_scope("copy"):
            ops = []
            #copy the encoder weights
            for w , w_other in zip(self.encoder_weights,other_enc.encoder_weights):
                ops.append(w.assign(w_other))
            for b , b_other in zip(self.encoder_biases, other_enc.encoder_biases):
                ops.append(b.assign(b_other))

            #copy the decoder weights
            for w , w_other in zip(self.decoder_weights,other_enc.decoder_weights):
                ops.append(w.assign(w_other))
            for b, b_other in zip(self.decoder_biases,other_enc.decoder_biases):
                ops.append(b.assign(b_other))

            #copy hidden weights
            for w , w_other in zip(self.hidden_weights,other_enc.hidden_weights):
                ops.append(w.assign(w_other))
            for b, b_other in zip(self.hidden_biases,other_enc.hidden_biases):
                ops.append(b.assign(b_other))

            return ops

    def list_weights(self):
        w = []
        w.extend(self.encoder_weights)
        w.extend(self.encoder_biases)
        w.extend(self.decoder_weights)
        w.extend(self.decoder_biases)
        w.extend(self.hidden_weights)
        w.extend(self.hidden_biases)
        return w

