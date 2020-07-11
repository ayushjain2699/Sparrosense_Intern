from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
import fastrand


from scipy import ndimage
import multiprocessing
from comput_motion_statistics_fast import pattern_1, pattern_2, pattern_3


new_height = 128
new_width = 171
crop_size= 112
clip_length=16

def read_batch(input):

    video_link, num_classes = input

    video_start_dir = video_link[0]
    video_end_dir = video_link[1]
    start_frame = int(video_link[3]) 
    end_frame = (video_link[4])
    path, dirs, files = next(os.walk(video_start_dir))
    ##########################################
    start_end = len(files)  #CONFIRM THIS NUMBER
    ##########################################
    if(end_frame == "-"):
        end_frame = start_end
    else
        end_frmae = int(end_frame)

    crop_x = random.randint(0, new_height - crop_size)  # crop size should be applied to all images
    crop_y = random.randint(0, new_width - crop_size)
    clip_sample_one = []

    if(video_start_dir == video_end_dir):

        if(end_frame - start_frame>=clip_length - 1):
            #start_frame = random.randint(start_frame,end_frame - clip_length + 1)
            start_frame = fastrand.pcg32bounded(end_frame-clip_length+1-start_frame+1)+start_frame


            for i in range(clip_length):
                cur_img_path = os.path.join(video_start_dir, "frame" + "{:06}.jpg".format(start_frame + i))
                img_origin = cv2.imread(cur_img_path)
                img_res = cv2.resize(img_origin, (171, 128))
                img = img_res.astype(np.float32)
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
                clip_sample_one.append(img)

        else:

            j = 0
            for i in range(clip_length):

                if(start_frame+j>end_frame):    j = 0

                cur_img_path = os.path.join(video_start_dir, "frame" + "{:06}.jpg".format(start_frame + j))
                img_origin = cv2.imread(cur_img_path)
                #clips_list.append(cur_img_path)
                img_res = cv2.resize(img_origin, (171, 128))
                img = img_res.astype(np.float32)
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
                clip_sample_one.append(img)
                j += 1


        clip_sample_one = np.array(clip_sample_one).astype(np.float32)  # 16 x 112 x 112 x 3

    else:
        total = start_end - start_frame + 1 + end_frame

        if(total>=clip_length):
            #start_frame = random.randint(start_frame,start_end + end_frame - clip_length + 1)
            start_frame = fastrand.pcg32bounded(start_end+end_frame-clip_length+1-start_frame+1)+start_frame


            for i in range(clip_length):

                if(start_frame + i>start_end): 
                    cur_img_path = os.path.join(video_end_dir, "frame" + "{:06}.jpg".format(start_frame + i - start_end))
                else:
                    cur_img_path = os.path.join(video_start_dir, "frame" + "{:06}.jpg".format(start_frame + i))   
                
                img_origin = cv2.imread(cur_img_path)
                #clips_list.append(cur_img_path)
                img_res = cv2.resize(img_origin, (171, 128))
                img = img_res.astype(np.float32)
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
                clip_sample_one.append(img)
        else:
            j = 0
            for i in range(clip_length):

                if(start_frame + j - start_end > end_frame): j = 0

                if(start_frame + j>start_end): 
                    cur_img_path = os.path.join(video_end_dir, "frame" + "{:06}.jpg".format(start_frame + j - start_end))
                else:
                    cur_img_path = os.path.join(video_start_dir, "frame" + "{:06}.jpg".format(start_frame + j))   
                
                img_origin = cv2.imread(cur_img_path)
                #clips_list.append(cur_img_path)
                img_res = cv2.resize(img_origin, (171, 128))
                img = img_res.astype(np.float32)
                img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
                clip_sample_one.append(img)
                j += 1

    class_label = int(video_link[2])-1

    one_hot_class_label = np.identity(num_classes)[class_label]

    label_sample_one = one_hot_class_label



    return clip_sample_one,label_sample_one


def read_all(video_filename, batch_size, num_classes, start_pos=-1,shuffle=True, cpu_num=12):


    batch_index = 0
    next_batch_start = -1

    train_clips = []
    label = []


    if start_pos < 0:
        shuffle=True

    if shuffle:
        video_indices = list(range(len(video_links)))
        random.shuffle(video_indices)  # shuffle index!
    else:
        video_indices = range(start_pos, len(video_links))

    links_batch = []      
    for index in video_indices:

        if (batch_index >= batch_size):  # get 30 samples
            next_batch_start = index
            break
        else:
            video_link = video_links[index].strip('\n').split()
            #print(rgb_line)
            links_batch.append((video_link,num_classes))
            batch_index = batch_index + 1

    data = (links_batch)
    p = multiprocessing.Pool(processes=cpu_num)

    results = p.map(read_batch, data)  # results: 16 x 8

    p.close()
    p.join()

    train_clips = []
    label = []
    for result in results:  # 30 x 16 x 112 x 112 x 3 label: 30 x 1
        sample_one, label_one = result
        train_clips.append(sample_one)
        label.append(label_one)

    np_train_clips = np.array(train_clips).astype(np.float32)  # N x 16 x 112 x 112 x 3
    np_arr_label = np.array(label).astype(np.float32)

    return np_train_clips, np_arr_label, next_batch_start