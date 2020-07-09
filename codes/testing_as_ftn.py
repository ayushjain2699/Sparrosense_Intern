import tensorflow as tf
import os
import numpy as np
import sys
sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import input_for_test
batch_size = 30
sess = tf.Session()
saver = tf.train.import_meta_graph('full_class_pred.ckpt-400.meta')

graph = tf.get_default_graph()
img_input = graph.get_tensor_by_name("Placeholder:0")
temp = graph.get_tensor_by_name("Placeholder_1:0")
temp_value = np.zeros([batch_size,14])
out_label = graph.get_tensor_by_name("fc2/out:0")
out_final = tf.nn.softmax(out_label)
y_target = graph.get_tensor_by_name("y_target:0")

correct_pred = tf.equal(tf.argmax(out_final,axis = 1),tf.argmax(y_target,axis = 1))
accuracy_tensor = tf.reduce_mean(tf.cast(correct_pred,"float"))

iterations = 50
rgb_list = '../list/test_rgb_5.list'
accuracy_vec = []
for i in range(1,3):
	ckpt = os.path.join(".", "full_class_pred.ckpt-{}".format(200*i))
	saver.restore(sess,ckpt)
	final_accuracy = 0
	for i in range(iterations):
	        test_images,test_labels,_ = input_for_test.read_all(rgb_filename = rgb_list,batch_size = batch_size,num_classes = 5,start_pos = -1,shuffle = True,cpu_num = 12)
	        # mse_loss = tf.reduce_mean(tf.squared_difference(out_label, y_target))
	        label_pred,label,accuracy = sess.run([out_label,out_final,accuracy_tensor],feed_dict = {img_input:test_images,y_target:test_labels,temp:temp_value})
	        final_accuracy = final_accuracy+accuracy

	print(final_accuracy/iterations)
	accuracy_vec.append(final_accuracy/iterations)

#print(test_labels)
#print(label_pred)
#print(label)
print(accuracy_vec)