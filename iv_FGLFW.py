# FR 1:1(verification) image version_FGLFW
import cv2
import numpy as np
import time
import mxnet as mx
import csv
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import model_loading as model
import face_cropping as crop
import face_recognition as recognition
import face_alignment as alignment

def input_arrange(line):
	line = line.rstrip() # 'Abel_Pacheco/Abel_Pacheco_0001.jpg'
	person = line.split('/')[0] # 'Abel_Pacheco'
	return person, line

def main(detect_model, recognition_model, line1, line2):

	# Recognition Speed Checking Start
	start_time = time.time()

	# person1, 2: Ground-Truth, path: image path
	person1, line1 = input_arrange(line1)
	person2, line2 = input_arrange(line2)
	if person1 == person2:
		gt_value = 1
	else:
		gt_value = 0
	path1 = 'lfw/{}'.format(line1)
	path2 = 'lfw/{}'.format(line2)

	# Image reading
	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)

	# Face detection
	bbox1, landmark1 = detect_model.detect(img1, threshold=0.8, scale=1.0)
	bbox2, landmark2 = detect_model.detect(img2, threshold=0.8, scale=1.0)

	# Face Cropping(ver0)
	face_img1 = crop.ver0(img1, bbox1, landmark1)
	face_img2 = crop.ver0(img2, bbox2, landmark2)

	# Face recognition - Similarity Result
	sim = recognition.verification(face_img1, face_img2, recognition_model)

	# Recognition Speed Checking End
	final_time = time.time() - start_time

	# (Optional) Object Detection Result Checking
	face_img = cv2.hconcat([face_img1, face_img2])
	cv2.imshow('detection_result', face_img)
	cv2.waitKey(1)
	# cv2.imwrite('fglfw_crop/{}_{}_{}.jpg'.format(idx,person1,person2), face_img)
	return sim, final_time, gt_value, face_img, person1, person2

if __name__ == "__main__":
	# Model Loading
	detect_model = model.detect_loading(0)
	recognition_model = model.recognition_loading(0)

	# Input
	input_file = 'label/pairs_FGLFW.txt'  # FGLFW pair file location
	i_f = open(input_file)

	idx = 0
	time_list = []
	img_list = []
    
	# crop ????????? ????????? ?????? ??????
	try:
		shutil.rmtree('fglfw_crop')
	except:
		pass
	os.makedirs('fglfw_crop')
    
	# Main Running
	while True:
		# Test Line Reading
		line1 = i_f.readline()
		line2 = i_f.readline()
		if (not line1) or (not line2):
			break

		# Main Code
		sim, final_time, gt_value, face_img, person1, person2 = main(detect_model, recognition_model, line1, line2)
		img_dict = {}
		img_dict['idx'] = idx
		img_dict['gt_value'] = gt_value
		img_dict['sim'] = sim
		img_dict['face_img'] = face_img
		img_dict['person1'] = person1
		img_dict['person2'] = person2
		img_list.append(img_dict)

		# ??????
		time_list.append(final_time)

		# HOW IS IT GOING?
		if idx % 100 == 0:
			print('{:2.2%} complete!'.format(idx / 6000))
		idx += 1

	i_f.close()

	# ??????
	avg_time = sum(time_list)/len(img_list)
	print("???????????? ?????? ?????? ??????: {:.1f}???".format(avg_time))

	# ?????? 
	sim_thr_list = [x/100 for x in range(0, 101)]

	# similarity threshold??? tar, trr, acc ????????? ????????? ??????
	tar_list = []
	trr_list = []
	acc_list = []
    
	# fglfw_result ??? ?????? ?????? ??? ???????????? ??????
	if os.path.exists('fglfw_result'):
		shutil.rmtree('fglfw_result')
	os.makedirs('fglfw_result')

	same_img_list = [img for img in img_list if img['gt_value'] == 1]
	diff_img_list = [img for img in img_list if img['gt_value'] == 0]
	same_idx_list = []
	diff_idx_list = []

	for sim_thr in sim_thr_list:

		same_cnt = 0
		diff_cnt = 0
		same_idxes = []
		diff_idxes = []

		for img in img_list:

			# GT ??? = 1 (same)
			if img['gt_value'] == 1:
				if img['sim'] >= sim_thr:
					same_cnt += 1
				else: # ?????? ????????? index ??????
					same_idx = img['idx']
					same_idxes.append(same_idx)

			# GT ??? = 0 (different)
			if img['gt_value'] == 0:
				if img['sim'] < sim_thr:
					diff_cnt += 1
				else: # ?????? ????????? index ??????
					diff_idx = img['idx']
					diff_idxes.append(diff_idx)

		tar = same_cnt/len(same_img_list)
		tar_list.append(tar)
		trr = diff_cnt/len(diff_img_list)
		trr_list.append(trr)
		acc = (same_cnt + diff_cnt) / (len(same_img_list) + len(diff_img_list))
		acc_list.append(acc)
		same_idx_list.append(same_idxes)
		diff_idx_list.append(diff_idxes)

	far_list = [1 - tar for tar in tar_list]
	frr_list = [1 - trr for trr in trr_list]

	auc = np.abs(np.trapz(y=trr_list, x=tar_list))
	print("AUC:", round(auc, 3))
	mean_acc = np.mean(acc_list)
	max_acc = np.max(acc_list)
	max_acc_thr = sim_thr_list[acc_list.index(max_acc)]
	print("?????? Accuracy:", round(mean_acc, 3))
	print("?????? Accuracy: %.3f (threshold: %.2f)" % (max_acc, max_acc_thr))

	# fglfw_result ??? ?????? Accuracy????????? threshold ?????? ??????
	os.mkdir('fglfw_result/{}'.format(max_acc_thr))

	# ?????? ????????? ??????
	for same_idx in same_idx_list[acc_list.index(max_acc)]:
		file_name = 'fglfw_result/{}/{}_{}_{}_{}.jpg'.format(max_acc_thr, same_idx, img_list[same_idx]['person1'], img_list[same_idx]['person2'], round(img_list[same_idx]['sim'], 4))
		cv2.imwrite(file_name, img_list[same_idx]['face_img'])
	for diff_idx in diff_idx_list[acc_list.index(max_acc)]:
		file_name = 'fglfw_result/{}/{}_{}_{}_{}.jpg'.format(max_acc_thr, diff_idx, img_list[diff_idx]['person1'], img_list[diff_idx]['person2'], round(img_list[same_idx]['sim'], 4))
		cv2.imwrite(file_name, img_list[diff_idx]['face_img'])
        
	# ?????? ????????? ????????? ??????
	with open('fglfw_result/result.txt', 'w') as f:
		f.write("<??????>\n")
		f.write("AUC: {}\n".format(round(auc, 3)))
		f.write("?????? Accuracy: {}\n".format(round(mean_acc, 3)))
		f.write("?????? Accuracy: %.3f (threshold: %.2f)\n" % (max_acc, max_acc_thr))
		f.write('\n')
		f.write("<??????>\n")
		f.write("???????????? ?????? ?????? ??????: {:.1f}???\n".format(avg_time))

	# ROC Curve ????????? ????????? 
	plt.scatter(tar_list, trr_list)
	plt.plot(tar_list, trr_list)
	plt.grid()
	plt.xlabel("TAR")
	plt.ylabel("TRR")
	plt.title("ROC Curve")
	plt.savefig('fglfw_result/ROC Curve.png')

	print('???!')