# 1-1-Face-Verification
> LFW와 FGLFW dataset에 대한 1:1 Face Verification

## LFW dataset

|**RetinaFace**|**ArcFace**|**Crop Version**|**AUC**|<p>**Average**</p><p>**Recognition**</p><p>**Speed**</p>|
| :-: | :-: | :-: | :-: | :-: |
|ResNet50|ResNet100|<strong>Ver1|<strong>0.989|1.1초|
|||Ver6|0.971|1.1초|
|||<p>Ver1</p><p>+Image rotation</p>|0.988|1.1초|
- 기존 모델에 대해 face cropping version을 바꿔가며 AUC, average recognition speed를 체크
- Face cropping ver1, ver6, ver1+Image rotation 중 <strong>ver1</strong>이 가장 AUC가 높음  
  <br>

|**RetinaFace**|**ArcFace**|**Crop Version**|**Accuracy**|**AUC**|<p>**Average**</p><p>**Recognition**</p><p>**Speed**</p>|
| :-: | :-: | :-: | :-: | :-: | :-: |
|<strong>MobileNet0.25|MobileFaceNet|Ver1|95.0%|0.989|0.2초|
||ResNet34||93.2%|0.977|0.4초|
||ResNet50||92.8%|0.977|0.5초|
||<strong>ResNet100||<strong>98.2%|<strong>0.996|<strong>0.9초|
|ResNet50|MobileFaceNet||95.4%|0.989|0.5초|
||ResNet34||93.0%|0.978|0.8초|
||ResNet50||92.5%|0.975|0.9초|
||ResNet100||98.0%|0.996|1.2초|
- Face cropping ver1으로 모든 모델에 대해 Accuracy, AUC, Average Recognition Speed를 체크
- Face Verification 결과 98.0% 이상의 Accuracy를 가지면서 속도가 1초 이내인 모델은 <strong>MobileNet0.25(RetinaFace), ResNet100(ArcFace)</strong>으로 확인됨
- MobileNet0.25는 [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)에서, MobileFaceNet, ResNet34, ResNet50은 [insightface](https://github.com/deepinsight/insightface/wiki/Model-Zoo)에서 다운가능  
<br>

|**RetinaFace**|**ArcFace**|**Crop Version**|**Accuracy**|**AUC**|<p>**Average**</p><p>**Recognition**</p><p>**Speed**</p>|
| :-: | :-: | :-: | :-: | :-: | :-: |
|MobileNet0.25|MobileFaceNet|ver1|95.0%|0.989|0.2초|
|||heuristic|97.3%|0.994||
|||<strong>ver0|<strong>99.37%|<strong>0.9992||
- 틀린 이미지를 확인하며 문제점을 보완해 새로운 face cropping ver0 생성
- MobileNet0.25(RetinaFace), MobileFaceNet(ArcFace)으로 crop version을 바꿔서 돌린 결과 ver0가 가장 accuracy가 높아 <strong>ver0으로 cropping 확정</strong>
- 참고로 ver0은 crop뿐만 아니라 allign도 포함되어 있음  
<br>
  
## FGLFW dataset

|**RetinaFace**|**ArcFace**|**Crop Version**|**Accuracy**|**AUC**|<p>**Average**</p><p>**Recognition**</p><p>**Speed**</p>|
| :-: | :-: | :-: | :-: | :-: | :-: |
|<strong>MobileNet0.25|MobileFaceNet|Ver1|88.0%|0.948|0.2초|
||ResNet34||85.9%|0.925|0.4초|
||ResNet50||86.0%|0.926|0.5초|
||<strong>ResNet100||<strong>96.4%|<strong>0.988|<strong>0.9초|
- FGLFW 데이터셋에 대해 Face Verification 결과 Accuracy가 가장 높으면서 속도가 1초 이내인 모델은 <strong>MobileNet0.25(RetinaFace), ResNet100(ArcFace)</strong>으로 확인됨
<br>

|**RetinaFace**|**ArcFace**|**Crop Version**|**Accuracy**|**AUC**|<p>**Average**</p><p>**Recognition**</p><p>**Speed**</p>|
| :-: | :-: | :-: | :-: | :-: | :-: |
|<strong>MobileNet0.25|MobileFaceNet|Ver0|97.6%|0.995|0.2초|
||ResNet34||99.0%|0.998|0.4초|
||ResNet50||99.2%|0.998|0.5초|
||<strong>ResNet100||<strong>99.6%|<strong>0.999|0.9초|
|ResNet50|MobileFaceNet||97.1%|0.993|0.5초|
||ResNet34||98.5%|0.997|0.8초|
||ResNet50||99.1%|0.997|0.9초|
||ResNet100||99.5%|0.998|1.2초|
- Face cropping ver0로 돌린 결과 Accuracy가 가장 높은 모델은 <strong>MobileNet0.25(RetinaFace), ResNet100(ArcFace)</strong>으로 확인됨
<br>
