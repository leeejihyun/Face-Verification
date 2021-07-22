import insightface_new
import numpy as np
import cv2
import time

# Model Loading : Deteciton and Recognition

def detect_loading(num):
	# face_detect_model = insightface_new.model_zoo.get_model('retinaface_r50_v1')
	face_detect_model = insightface_new.model_zoo.get_model('retinaface_mnet025_v2')
	face_detect_model.prepare(ctx_id = num, nms=0.4)
	return face_detect_model

def recognition_loading(num):
	# face_recognition_model = insightface_new.model_zoo.get_model('arcface_r100_v1')
	# face_recognition_model = insightface_new.model_zoo.get_model('arcface_r50_v1')
	# face_recognition_model = insightface_new.model_zoo.get_model('arcface_r34_v1')
	face_recognition_model = insightface_new.model_zoo.get_model('arcface_mfn_v1')
	face_recognition_model.prepare(ctx_id = num)
	return face_recognition_model