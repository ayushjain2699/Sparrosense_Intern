import tensorflow as tf

def fc(input,output_dims,regularizer = True, dropout = False):
	
	if regularizer: regularizer = tf.contrib.layers.l2_regularizer(0.005)

	input_dims = input.get_shape().as_list()

    with tf.variable_scope('fc1'):
        fc1_weight = tf.get_variable('train_weight', [input_dims[1], 2048],initializer=tf.random_normal_initializer(stddev=0.005))
        if regularizer:
            tf.add_to_collection("weight_decay_loss", regularizer(fc1_weight))
        fc1_bias = tf.get_variable('train_bias', [2048], initializer=tf.constant_initializer(1.0))
        fc1 = tf.nn.relu(tf.matmul(input, fc1_weight) + fc1_bias)
        if dropout:
            fc1 = tf.nn.dropout(fc6, 0.5)	

    with tf.variable_scope('fc2'):
	    out_weight = tf.get_variable('train_weight', [2048, output_dims], initializer=tf.random_normal_initializer(stddev=0.01))
	    if regularizer:
	        tf.add_to_collection("weight_decay_loss", regularizer(out_weight))
	    out_bias = tf.get_variable('train_bias', [output_dims], initializer=tf.constant_initializer(0.0))
	    out1 = tf.matmul(fc1, out_weight)
	    out = tf.add(out1, out_bias, name = 'out')

	return out