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
iterations = 3000
iterations_for_accuracy = 20
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

	opt_weight = tf.train.MomentumOptimizer(lr_weight, momentum=momentum,name = 'momentum2')
	opt_bias = tf.train.MomentumOptimizer(lr_bias, momentum=momentum,name = 'momentum2')

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

	out_final = tf.nn.softmax(y_pred)
	correct_pred = tf.equal(tf.argmax(out_final,axis = 1),tf.argmax(y_target,axis = 1))
	accuracy_tensor = tf.reduce_mean(tf.cast(correct_pred,"float"))

	rgb_list = 'list/rgb_5.list'
	init = tf.global_variables_initializer()
	sess.run(init)
	train_writer = tf.summary.FileWriter('./visual_logs/train_full_class_pred', sess.graph)
	total_accuracy = 0
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

		feature, summary, loss_value, ce_loss, _, old_weight,accuracy= sess.run([
		    reshaped, merged, loss, softmax_loss, train_op, grad_weight,accuracy_tensor], feed_dict={
		    img_input: train_images,
		    y_target: train_labels,
		    temp: temp_value
		})
		total_accuracy = total_accuracy+accuracy
        if i % (display) == 0:
            print("softmax_loss:", ce_loss)
            print("loss:", loss_value)
            train_writer.add_summary(summary, i)
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (i, duration))


        if i % 200 == 0:

        	print("Avg accuracy from step %d to %d: %.3f" % (i-200,i,total_accuracy/200))
        	total_accuracy = 0

        	final_accuracy = 0
        	for i in range(iterations_for_accuracy):
		        test_images,test_labels,_ = new_input_data.read_all(rgb_filename = rgb_list,batch_size = batch_size,num_classes = num_classes,start_pos = -1,shuffle = True,cpu_num = cpu_num)
		        label_pred,label,accuracy = sess.run([y_pred,out_final,accuracy_tensor],feed_dict = {img_input:test_images,y_target:test_labels,temp:temp_value})
		        final_accuracy = final_accuracy+accuracy

        	print("Accuracy at step %d: %.3f" % (i,final_accuracy/iterations_for_accuracy))

            saver.save(sess, os.path.join("full_class_pred_model", model_name), global_step=global_step)


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()