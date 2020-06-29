import tensorflow as tf
import os
import model_class_det
import time
import new_input_data
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

display = 1
num_classes = 5
batch_size = 30
iterations = 4000
base_lr = 0.001
momentum = 0.9
cpu_num = 12
model_name = "full_class_pred.ckpt"

def train():

	sess = tf.Session()
	saver = tf.train.import_meta_graph('motion_pattern_all_new_global/model_motion_statistics.ckpt-4000.meta')
	#saver.restore(sess,tf.train.latest_checkpoint('motion_pattern_all_new_global/'))

	graph = tf.get_default_graph()
	img_input = graph.get_tensor_by_name("Placeholder:0")
	reshaped = graph.get_tensor_by_name("C3D/reshaped:0")
	temp = graph.get_tensor_by_name("Placeholder_1:0")
	temp_value = np.zeros([batch_size,14])
	#reshaped = tf.stop_gradient(reshaped)
	y_target = tf.placeholder(tf.float32, shape=(None,num_classes),name = "y_target")
	# sys.path.append('../')

	y_pred = model_class_det.fc(reshaped, num_classes, regularizer = True, dropout = False)

	global_step = tf.Variable(0, trainable=False)

	varlist_weight = []
	varlist_bias = []
	trainable_variables = tf.trainable_variables()
	for var in trainable_variables:
	    if 'weight' in var.name:
	        varlist_weight.append(var)
	    elif 'bias' in var.name:
	        varlist_bias.append(var)

	lr_weight = tf.train.exponential_decay(base_lr, global_step, 1000, 0.1,staircase=True)  
	lr_bias = tf.train.exponential_decay(base_lr * 2, global_step, 1000, 0.1,staircase=True)

	opt_weight = tf.train.MomentumOptimizer(lr_weight, momentum=momentum)
	opt_bias = tf.train.MomentumOptimizer(lr_bias, momentum=momentum)

	softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pred,labels = y_target))

	weight_decay_loss = tf.add_n(tf.get_collection('weight_decay_loss'))

	loss = softmax_loss + weight_decay_loss	
	tf.summary.scalar('softmax_loss', softmax_loss)
	tf.summary.scalar('weight_decay_loss', weight_decay_loss)
	tf.summary.scalar('total_loss', loss)

	grad_weight = opt_weight.compute_gradients(loss, varlist_weight)
	grad_bias = opt_bias.compute_gradients(loss, varlist_bias)
	apply_gradient_op_weight = opt_weight.apply_gradients(grad_weight)
	apply_gradient_op_bias = opt_bias.apply_gradients(grad_bias, global_step=global_step)
	train_op = tf.group(apply_gradient_op_weight, apply_gradient_op_bias)

	saver = tf.train.Saver()
	merged = tf.summary.merge_all()

	rgb_list = 'list/rgb_5.list'

	init = tf.global_variables_initializer()
	sess.run(init)

	train_writer = tf.summary.FileWriter('./visual_logs/train_full_class_pred', sess.graph)

    for i in range(1,iterations+1):
        start_time = time.time()

		train_images, train_labels, next_batch_start = new_input_data.read_all(
		    rgb_filename=rgb_list,
		    batch_size=batch_size,
		    num_classes = num_classes,
		    start_pos=-1,
		    shuffle=True,
		    cpu_num=cpu_num      
		)

        duration = time.time() - start_time
        print('read data time %.3f sec' % (duration))

		feature, summary, loss_value, ce_loss, _, old_weight = sess.run([
		    reshaped, merged, loss, softmax_loss, train_op, grad_weight], feed_dict={
		    img_input: train_images,
		    y_target: train_labels,
		    temp: temp_value
		})

        if i % (display) == 0:
            print("softmax_loss:", ce_loss)
            print("loss:", loss_value)
            train_writer.add_summary(summary, i)
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (i, duration))


        if i % 1000 == 0:
            saver.save(sess, os.path.join("full_class_pred_model", model_name), global_step=global_step)

def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()