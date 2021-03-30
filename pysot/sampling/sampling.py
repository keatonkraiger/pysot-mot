import numpy as np

def exhaustive_samp(img, bbox):
	img_w, img_h, _ = img.shape()
	rois = []
	for bb in bbox:
		cx = bbox[0] + self.center_pos[0]
		cy = bbox[1] + self.center_pos[1]
		# smooth bbox
		width = self.size[0] * (1 - lr) + bbox[2] * lr
		height = self.size[1] * (1 - lr) + bbox[3] * lr
		for i in range(0, img_w-width):
			for j in range(0, img_h-height):
				rois.append([cx_i,cy+j,width,height])
	return rois


def random_samp(img, bbox, window_ratio, num_sample_per_bb):
	img_w, img_h, _ = img.shape()
	rois = []
	window_size_w = window_ratio*img_w
	window_size_h = window_ratio*img_h
	for bb in bbox:
		cx = bbox[0] + self.center_pos[0]
		cy = bbox[1] + self.center_pos[1]
		# smooth bbox
		width = self.size[0] * (1 - lr) + bbox[2] * lr
		height = self.size[1] * (1 - lr) + bbox[3] * lr
		for i in range(0,num_sample_per_bb):
			cx_i = np.random.randint( max(0,cx-np.int(window_size_w/2)), min(img_w,cx+np.int(window_size_w/2)+width), size=1)
			cy_i = np.random.randint( max(0,cy-np.int(window_size_h/2)), min(img_h,cy+np.int(window_size_h/2)+height), size=1)
			rois.append([cx_i,cy_i,width,height])
	return rois


def mcmc_samp():
	return