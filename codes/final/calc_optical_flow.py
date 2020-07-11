import numpy
import cv2
import os

def calc_flow(video_link):

	path, dirs, files = next(os.walk(video_link))
	count = len(files)

    os.mkdir(video_link+"/"+"u_flow")
	os.mkdir(video_link+"/"+"v_flow")
	
	cur_img_path = os.path.join(video_link, "frame" + "{:06}.jpg".format(1))
	img_origin = cv2.imread(cur_img_path)
	img_res = cv2.resize(img_origin, (171, 128))
	frame1 = img_res

	for i in range(2,count+1):

		cur_img_path = os.path.join(video_link, "frame" + "{:06}.jpg".format(i))
		img_origin = cv2.imread(cur_img_path)
		img_res = cv2.resize(img_origin, (171, 128))
		frame2 = img_res

		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		nex = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

		flow = cv2.calcOpticalFlowFarneback(prvs, nex,None, 0.5, 3, 15, 3, 5, 1.2, 0)

		# Change here
		horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
		vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
		horz = horz.astype('uint8')
		vert = vert.astype('uint8')
		
		print("extracting frame = %d"%(i-1))
		cv2.imwrite(os.path.join(video_link,"u_flow","frame{:06}.jpg".format(i-1)), horz)
		cv2.imwrite(os.path.join(video_link,"v_flow","frame{:06}.jpg".format(i-1)), vert)
		frame1 = frame2



video_filename = "list_SSL_home_dir.list"

video_links = open(video_filename, 'r')
video_links = list(video_links)

for i in range(len(video_links)):
	print("calculating flow of video : %d"%(i+1))
	video_link = video_links[i].strip('\n')
	calc_flow(video_link)
