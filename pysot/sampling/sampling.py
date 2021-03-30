import numpy as np

def exhaustive_samp(img, bbox):
	img_w, img_h, _ = img.shape
	rois = []
	for bb in bbox:
		cx = bbox[0] #+ self.center_pos[0]
		cy = bbox[1] #+ self.center_pos[1]
		# smooth bbox
		width = bbox[2] #self.size[0] * (1 - lr) + bbox[2] * lr
		height = bbox[3] #self.size[1] * (1 - lr) + bbox[3] * lr
		for i in range(0, img_w-width):
			for j in range(0, img_h-height):
				rois.append([cx_i,cy+j,width,height])
	return rois


def random_samp(img, bbox, window_ratio, num_sample_per_bb):
	img_w, img_h, _ = img.shape
	rois = []
	window_size_w = window_ratio*img_w
	window_size_h = window_ratio*img_h
	# lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
	for bb in bbox:
		cx = bbox[0] #+ self.center_pos[0]
		cy = bbox[1] #+ self.center_pos[1]
		# smooth bbox
		width = bbox[2] #self.size[0] * (1 - lr) + bbox[2] * lr
		height = bbox[3] #self.size[1] * (1 - lr) + bbox[3] * lr
		for i in range(0,num_sample_per_bb):
			cx_i = np.random.randint( max(0,cx-np.int(window_size_w/2)), min(img_w,cx+np.int(window_size_w/2)+width), size=1)
			cy_i = np.random.randint( max(0,cy-np.int(window_size_h/2)), min(img_h,cy+np.int(window_size_h/2)+height), size=1)
			rois.append([cx_i,cy_i,width,height])
	return rois


def roi_norm(cx,cy,width,height,velo,std):
	rois = []
	# x_vals = np.random.normal(cx+velo[0],std[0], 10)
	# y_vals = np.random.normal(cy+velo[1],std[1], 10)
	mean = np.array([cx,cy])
	cov = np.array([[std[0],0],[0,std[0]]])
	x_vals, y_vals = np.random.multivariate_normal(mean, cov, 10)
	for i in range(0,x_vals.shape[0]):
		rois.append([x_vals[i], y_vals[i], width, height])
	return rois 


def velo_norm_samp(img, bbox, velo, std_dev):
	img_w, img_h, _ = img.shape
	rois = []
	for m in range(0,len(bbox)):
		cx = bbox[0] #+ self.centre_pos[0]
		cy = bbox[1] #+ self.centre_pos[1]
		width = bbox[2]
		height = bbox[3]
		temp_rois = noi_norm(cx,cy,width,height,velo[i],std_dev[i])
		rois.extend(temp_rois)
	return rois


