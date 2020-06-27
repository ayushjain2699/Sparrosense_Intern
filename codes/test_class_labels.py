import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 30
iterations = 20
accuracy_final = 0

import sys
sess = tf.Session()
saver = tf.train.import_meta_graph('class_pred.ckpt-4000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
img_input = graph.get_tensor_by_name("Placeholder:0")
temp = graph.get_tensor_by_name("Placeholder_1:0")
temp_value = np.zeros([batch_size,14]) 
out_label = graph.get_tensor_by_name("fc2/out:0")
out_final = tf.nn.softmax(out_label)
y_target = graph.get_tensor_by_name("y_target:0")

correct_pred = tf.equal(tf.argmax(out_label,1),tf.argmax(y_target,1))
accuracy_tensor = tf.reduce_mean(tf.cast(correct_pred,"float"))

sys.path.append('../')
import new_input_data

rgb_list = '../list/test_rgb_5.list'
for i in range(iterations):
	test_images,test_labels,_ = new_input_data.read_all(rgb_filename = rgb_list,batch_size = batch_size,num_classes = 5,start_pos = -1,shuffle = True,cpu_num = 12)
	# mse_loss = tf.reduce_mean(tf.squared_difference(out_label, y_target))
	label_pred,accuracy = sess.run([out_final,accuracy_tensor],feed_dict = {img_input:test_images,y_target:test_labels,temp:temp_value})
	accuracy_final = accuracy_final + accuracy

accuracy_final = accuracy_final/iterations
#print(test_labels)
#print(label_pred)
print(accuracy_final)