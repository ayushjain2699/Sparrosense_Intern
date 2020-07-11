import tensorflow as tf
import os
import numpy as np
import input_SL_orig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 30
import sys
sess = tf.Session()
saver = tf.train.import_meta_graph('./SL_Model/SL.ckpt-3000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./SL_Model/'))

graph = tf.get_default_graph()
img_input = graph.get_tensor_by_name("img_input:0")
temp = graph.get_tensor_by_name("y_target:0")
temp_value = np.zeros([batch_size,14])
out_label = graph.get_tensor_by_name("fc2/out:0")
out_final = tf.nn.softmax(out_label)
y_target = graph.get_tensor_by_name("y_target_SL:0")

correct_pred = tf.equal(tf.argmax(out_final,axis = 1),tf.argmax(y_target,axis = 1))
accuracy_tensor = tf.reduce_mean(tf.cast(correct_pred,"float"))

iterations = 20
video_list = 'list/test_SL_list.list'
final_accuracy = 0
for i in range(iterations):
        test_images,test_labels,_ = input_SL_orig.read_all(video_filename = video_list,batch_size = batch_size,num_classes = 5,start_pos = -1,shuffle = True,cpu_num = 12)
        # mse_loss = tf.reduce_mean(tf.squared_difference(out_label, y_target))
        label_pred,label,accuracy = sess.run([out_label,out_final,accuracy_tensor],feed_dict = {img_input:test_images,y_target:test_labels,temp:temp_value})
        final_accuracy = final_accuracy+accuracy

#print(test_labels)
#print(label_pred)
#print(label)
print(final_accuracy/iterations)
