import insightface
import numpy as np
import cv2
from skimage import transform as trans

def ver0(img, bbox, landmark):
    height, width = img.shape[0], img.shape[1]
    
    if len(bbox) == 1:
        face_img = ver5(img, bbox)
        face_img = face_rotate(face_img, bbox[0], landmark[0])
    elif len(bbox) == 0:
        print('wierd case happen')
        face_img = img[10:width-10, 10:width-10]
    elif len(bbox) > 1:
        #face_img, bbox2, landmark2 = ver6(img, bbox, landmark)
        face_img, bbox2, landmark2 = ver1(img, bbox, landmark)
        face_img = face_rotate(face_img, bbox2, landmark2)
    return face_img

def face_rotate(img, bbox, landmark):
    M = None
    src = np.array([
        [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
        [33.5493, 92.3655], [62.7299, 92.2041] ], dtype=np.float32)
    src[:,0] *= 112/96
    src[:,1] -= 4
    dst = landmark_transform(bbox, landmark)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img1 = cv2.warpAffine(img, M, (112, 112), borderValue = 0.0)
    return face_img1

def landmark_transform(bbox, landmark):
    bbox = bbox.flatten()
    x0 = bbox[0]
    y0 = bbox[1]
    for i in range(5):
        landmark[i][0] = landmark[i][0] - x0
        landmark[i][1] = landmark[i][1] - y0
    box_len0 = bbox[2]-bbox[0]
    box_len1 = bbox[3]-bbox[1]
    for i in range(5):
        landmark[i][0] /= box_len0/112
        landmark[i][1] /= box_len1/112
    dst = landmark.astype(np.float32)
    return dst

def ver1(img, bboxs, landmark):
    #bbox = bbox.astype(np.int).flatten()
    expand_ratio = -0.01
    f_box= [0,0,0,0]
    b_idx = 0
    height, width = img.shape[0], img.shape[1]

    for i, bbox in enumerate(bboxs):
        if bbox[0]<width/2 and bbox[1]<height/2 and bbox[2]>width/2 and bbox[3]>height/2 :
            center_x = int(0.5 * (bbox[0] + bbox[2]))
            center_y = int(0.5 * (bbox[1] + bbox[3]))
            length = int(0.5 * (1 + expand_ratio) * (bbox[3] - bbox[1]))
            f_box = [center_x - length, center_y - length, center_x + length, center_y + length]
            face_img = img[max(1, f_box[1]):f_box[3], max(1, f_box[0]):f_box[2]]
            face_img = cv2.resize(face_img, (112, 112))
            return face_img, bboxs[i], landmark[i]
            break
    if f_box == [0, 0, 0, 0]:
        print('Face is not centered or not detected')
        center_x = int(0.5 * width)
        center_y = int(0.5 * height)
        length = int(0.25 * height)
        
        f_box = [center_x - length, center_y - length, center_x + length, center_y + length]
        f_box = np.array(f_box)
        f_box[f_box<0]=0
        if f_box[2]>w:
            f_box[2]=w
        if f_box[3]>h:
            f_box[3]=h
        face_img = img[max(1, f_box[1]):f_box[3], max(1, f_box[0]):f_box[2]]
        face_img = cv2.resize(face_img, (112, 112))
        return face_img

def ver5(img, bbox):
    h, w = img.shape[:2]
    bbox = bbox.astype(np.int).flatten()
    center_x = int(0.5 * (bbox[0]+bbox[2]))
    center_y = int(0.5 * (bbox[1]+bbox[3]))
    length = int(0.5 * (bbox[3]-bbox[1]))
    f_box = [center_x-length, center_y-length, center_x+length, center_y+length]
    if f_box[0]<0 : f_box[0] = 1
    if f_box[1]<0 : f_box[1] = 1
    if f_box[2]>w : f_box[2] = w-1
    if f_box[3]>h : f_box[3] = h-1
    face_img = img[f_box[1]:f_box[3], f_box[0]:f_box[2]]
    face_img = cv2.resize(face_img, (112, 112))
    return face_img

def ver6(img, bboxs, landmark, expand_ratio=-0.01):
    h, w = img.shape[:2]
    img_center = np.array([img.shape[1]/2, img.shape[0]/2])
    center_x =(bboxs[:,0]+bboxs[:,2])/2
    center_y = (bboxs[:,1]+bboxs[:,3])/2
    center_bbox = np.vstack([center_x, center_y]).T
    l2_dists = np.sqrt(np.square(img_center-center_bbox).sum(axis=1))
    
    length_w = (bboxs[:,2]-bboxs[:,0])
    length_h = (bboxs[:,3]-bboxs[:,1])
    areas = length_w*length_h
    
    confidences = bboxs[:,4]
    
    eval_bbox = np.argsort(-l2_dists)*0.3 + np.argsort(areas)*0.5 + np.argsort(confidences)*0.2
    idx = np.argmax(eval_bbox)
    
    new_x1 = center_x[idx] - (length_w[idx]*(1+expand_ratio)/2)
    new_x2 = center_x[idx] + (length_w[idx]*(1+expand_ratio)/2)
    new_y1 = center_y[idx] - (length_h[idx]*(1+expand_ratio)/2)
    new_y2 = center_y[idx] + (length_h[idx]*(1+expand_ratio)/2)
    
    if new_x1<0: new_x1 = 1
    if new_y1<0: new_y1 = 1
    if new_x2>w: new_x2 = w-1
    if new_y2>h: new_y2 = h-1
        
    face_img = img[int(new_y1):int(new_y2), int(new_x1):int(new_x2), :]
    return cv2.resize(face_img, (112, 112)), bboxs[idx], landmark[idx]