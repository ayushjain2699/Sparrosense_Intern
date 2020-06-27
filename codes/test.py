import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sess = tf.Session()
saver = tf.train.import_meta_graph('model_motion_statistics.ckpt-4000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
img_input = graph.get_tensor_by_name("Placeholder:0")
y_target = graph.get_tensor_by_name("Placeholder_1:0")
out_label = graph.get_tensor_by_name("C3D/fc8/out:0")

sys.path.append('../')
import input_data

rgb_list = '../list/rgb_5.list'
u_flow_list = '../list/u_flow_5.list'
v_flow_list = '../list/v_flow_5.list'

batch_size = 5
test_images,test_labels,_ = input_data.read_all(rgb_filename = rgb_list,u_flow_filename = u_flow_list,v_flow_filename = v_flow_list,batch_size = batch_size,start_pos = -1,shuffle = True,cpu_num = 12)
mse_loss = tf.reduce_mean(tf.squared_difference(out_label, y_target))
label_pred,loss = sess.run([out_label,mse_loss],feed_dict = {img_input:test_images,y_target:test_labels})