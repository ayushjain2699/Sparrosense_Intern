import os
import cv2
import pandas as pd
import time

def extractFrames(pathIn):
    direc = pathIn[68:78]
    name = pathIn[79:134]
    if(os.path.exists(direc+"/"+name)):
        print("Video already extracted")
    else:
        os.mkdir(direc+"/"+name)
        cap = cv2.VideoCapture(pathIn)
        count = 1
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
                print('Read %d frame: ' % count, ret)
                cv2.imwrite(os.path.join(direc,name,"frame{:06}.jpg".format(count)), frame)  # save frame as JPEG file
                count += 1
            else:
                break


        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
 

list_of_videos = "unique_videos_test.list"
l = open(list_of_videos,'r')
l = list(l)

print("Extracting RGB frames....")

for i in range(len(l)):
    print("Extracting video number : %d"%i)
    #start_time = time.time()
    pathIn = l[i].strip('\n')

    extractFrames(pathIn = pathIn)
    #duration = time.time() - start_time
    #print('extracting frames time %.3f sec' % (duration))
    #df = df.append({'video':pathIn,'count':count},ignore_index = True)

#df.to_csv(r'/home/neo/data/Ayush/rgb_orig_videos/df.csv', index = False, header=True)
