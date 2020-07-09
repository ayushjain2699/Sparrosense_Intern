import os
import cv2
import pandas as pd
import time

def extractFrames(video_list):
    path_start = video_list[0]
    path_end = video_list[1]
    start_frame = video_list[3]
    end_frame = video_list[4]
    direc_start = path_start[68:78]
    name_start = path_start[79:134]
    direc_end = path_end[68:78]
    name_end = path_end[79:134]
    if(not os.path.exists(direc_start+"/"+name_start)):
        os.mkdir(direc_start+"/"+name_start)
    if(not os.path.exists(direc_end+"/"+name_end)):
        os.mkdir(direc_end+"/"+name_end)

    end_count = 0

    if(path_start == path_end):
        cap = cv2.VideoCapture(path_start)
        count = 1
        if(end_frame == "-"):
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()

                if (ret == True and count<start_frame):
                    count += 1

                elif (ret == True and count>=start_frame):
                    #print('Read %d frame: ' % count, ret)
                    cv2.imwrite(os.path.join(direc_start,name_start,"frame{:06}.jpg".format(count)), frame) 
                    count += 1 # save frame as JPEG file

                else:
                    end_count = count
                    break

        else:
            while(cap.isopened()):
                ret, frame = cap.read()

                if (ret == True and count<start_frame):
                    count += 1

                elif ret == True and count>=start_frame and count<=end_frame:
                    #print('Read %d frame: ' % count, ret)
                    cv2.imwrite(os.path.join(direc_start,name_start,"frame{:06}.jpg".format(count)), frame) 
                    count += 1 # save frame as JPEG file
                else:
                    break

    else:
        cap = cv2.VideoCapture(path_start)
        count = 1
            while (cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if (ret == True and count<start_frame):
                    count += 1

                elif ret == True and count>=start_frame:
                    #print('Read %d frame: ' % count, ret)
                    cv2.imwrite(os.path.join(direc_start,name_start,"frame{:06}.jpg".format(count)), frame)
                    count += 1  # save frame as JPEG file
                else:
                    break

        cap = cv2.VideoCapture(path_end)
        count = 1

        start_frame = 1
            while(cap.isOpened()):
                ret, frame = cap.read()

                if ret == True and count<=end_frame:
                    #print('Read %d frame: ' % count, ret)
                    cv2.imwrite(os.path.join(direc_end,name_end,"frame{:06}.jpg".format(count)), frame)
                    count += 1  # save frame as JPEG file
                else:
                    break



    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return end_count;


list_of_videos = "list.list"
l = open(list_of_videos,'r')
l = list(l)

df = pd.DataFrame(columns=['video','count'])

for i in range(len(l)):

    video_list = l[i].strip('\n').split()
    print("list number : %d"%i)
    start_time = time.time()
    count = extractFrames(video_list = video_list)
    duration = time.time() - start_time
    print('extracting frames time %.3f sec' % (duration))
    if(count > 0):
        df = df.append({'video':video_list[0],'count':count},ignore_index = True)

df.to_csv(r'/home/neo/data/Ayush/df.csv', index = False, header=True)