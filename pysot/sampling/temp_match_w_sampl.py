import cv2 as cv
import numpy as np 
import os
import sampling as smpl

folder_name = '/home/anerudh/pennstate_sp21/ee554_computervision2/project/pysot-mot/trackdat-master/dl/vot18smaller/vot18smaller/zebrafish1'
gt_txt = 'groundtruth.txt'

image_files = [f for f in os.listdir(folder_name) if f.endswith('.jpg')]
image_files.sort()

gt_file = os.path.join(folder_name,gt_txt)

f = open(gt_file,'r')
img = cv.imread(os.path.join(folder_name,image_files[0]))
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sim_thresh = 0.7

#These lines below prepare the bounding box from the groundtruth file
line = f.readline()[:-1].split(',')
lin = []
for a in line:
	lin.append(int(float(a)))
# print(lin)
top = min(lin[1],lin[3],lin[5],lin[7])
left = min(lin[0],lin[2],lin[4],lin[6])
bottom = max(lin[1],lin[3],lin[5],lin[7])
right = max(lin[0],lin[2],lin[4],lin[6])
bbox = np.array([top,left,bottom,right])

#template
tmpl = img[top:bottom,left:right]
# tmp_hist = cv.calcHist([tmpl],[0],None, [256], [0,256])
tmp_hist = cv.calcHist([tmpl],[0,1,2],None, [256,256,256], [0,256,0,256,0,256])
tmp_hist = cv.normalize(tmp_hist,tmp_hist).flatten()

sample_method = 'EXHAUSTIVE'
# sample_method = 'RANDOM'
# sample_method = 'VELO_ADAPT'


for i in range(1,2): #len(image_files)):
	#The lines below read the next image to sample from 
	img2 = cv.imread(os.path.join(folder_name,image_files[i]))
	# img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	if(sample_method == 'EXHAUSTIVE'):
		rois = smpl.exhaustive_samp(img2,bbox)
	elif(sample_method == 'RANDOM'):
		window_ratio = 0.3
		samples_per_bbox = 10
		rois = smpl.random_samp(img2,bbox,window_ratio,samples_per_bbox)
	elif(sample_method == 'VELO_ADAPT'):
		velo = []
		rois = smpl.velo_norm_samp(img2, bbox, velo, std_dev)
	# print('ALL ROIs:',rois)
	result_roi = []
	sim = []
	best_sim_val = 0.0
	# print('ROIs that cleared the sim_thresh:')
	for roi in rois:
		test = img2[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],:]
		# test_hist = cv.calcHist([test], [0], None, [256], [0,256])
		test_hist = cv.calcHist([test], [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
		test_hist = cv.normalize(test_hist,test_hist).flatten()
		sim_val = cv.compareHist(test_hist,tmp_hist, cv.HISTCMP_BHATTACHARYYA)
		sim.append(sim_val)
		if(sim_val >= sim_thresh):
			# print(roi)
			if(best_sim_val<sim_val):
				best_sim_val = sim_val
				best_roi = roi
				best_hist = test_hist
	bbox = best_roi
	tmp_hist = best_hist
	print("Best matched ROI and similarity:")
	print(best_roi, ':', best_sim_val)	
	# best_test = img2[best_roi[1]:best_roi[1]+best_roi[3],best_roi[0]:best_roi[0]+best_roi[2]]
	# cv.imwrite('/home/anerudh/Pictures/best_'+str(i)+'.jpg', best_test)
	print('END')








