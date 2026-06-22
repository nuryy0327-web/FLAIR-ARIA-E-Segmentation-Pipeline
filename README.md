# FLAIR 기반 ARIA-E Segmentation Pipeline

코랩에서 작성하였습니다. ipynb 파일 용량이 높아지며 github에 업로드하지 못하는 문제가 있었어서, 전처리 및 파라미터 튜닝/결과 확인을 위한 시각화 과정은 대부분 주석처리 해놓은 상태입니다. 

그럼에도 용량이 높아 github에서 Unable to render code block 오류가 뜰 수 있습니다. -> 구글드라이브 링크에 동일파일 첨부해두었습니다.

모두 돌아가는데 그대로 돌릴 경우 3~4분, 주석을 모두 풀고 돌릴 경우 10분정도 걸립니다.(모델학습 파트 제외)

**현재 2.5D 기반의 모델링으로 변경한 후 모델학습은 돌리지 않은 상황입니다. 다만, inference 수행을 위해 기존의 학습된 2D 모델을 그대로 호출해 가중치 채널을 3개로 복제해 사용했습니다. 

## 실행 방법

코랩에서 실행하는것을 추천드립니다.(목차 확인 용이, 시각화가 github에서 잘리는 현상)  

해당 코드 및 기존 데이터 폴더와 첨부된 `checkpoint_classifier.pth`, `checkpoint_edema_classifier.pth`, `checkpoint.pth`을 구글 드라이브에 업로드한 상태에서 그대로 모두 실행하시면 됩니다 (코랩 환경 기준, 파일 저장 경로 : /content/drive/MyDrive).

> 모델 학습 시간이 오래 걸려 주석 처리해놓았습니다. 구글 드라이브에서 저장된(2D) 모델을 불러오게 되어있으니 주석 제거 없이 실행하시면 됩니다.

조금 더 시간을 단축하기 위해선, 첨부된 `flair_volumes`, `dicoms_preprocessed`, `results`,`results_gif` 폴더도 구글 드라이브에 업로드 후 실행하시면 좋습니다. (업로드하지 않고 실행하면 알아서 새로 폴더를 생성합니다.)

https://drive.google.com/drive/folders/1-nKTMeC0kuSxZbeUwSiKyift03l4wCWf?usp=sharing  
(구글드라이브 링크에 모두 업로드되어있습니다. 공유폴더 바로가기를 /content/drive/MyDrive에 두면 업로드 생략 가능할 것 같습니다.)

### 파이프라인 순서

1. 데이터 탐색 + 3D FLAIR image volume 구성 + sagittal study에서 axial slice 추출
2. 전처리(1단계 모델 방법론)
3. 학습용 Reference mask 생성
4. 모델 구성 (CNN Classifier / EDEMA Classifier / U-Net)
5. 각 모델 학습 또는 checkpoint 로드
6. Inference (3단계 파이프라인)
7. 3D mask volume 구성 및 후처리
8. 결과 저장 및 시각화

---
## DICOM 데이터 처리 방식
study별 FLAIR image volume[slice, row, column]을 구성해 사용하였습니다. 

### 촬영 방향 판별 및 통일

제공 데이터에는 Axial FLAIR와 Sagittal FLAIR가 혼재되어 있습니다. 2D 모델에서는 한 장의 image가 동일한 `[row, column]` 형태로 들어가더라도, Axial과 Sagittal에서 row/column이 의미하는 해부학적 방향이 다르기 때문에 하나의 모델에 혼재하여 입력하면 문제가 발생합니다.

이를 해결하기 위해 DICOM 메타데이터의 `ImageOrientationPatient` 태그를 활용하여 촬영 방향을 판별하고, Sagittal volume을 Axial 방향으로 변환하여 모든 데이터를 Axial 기준으로 통일하였습니다.

(데이터의 양을 확보하기 위해 방향별 두개의 모델이 아닌 하나의 모델을 사용하기로 했습니다.)

---

### sagittal slice -> axial slice 변환 방법

우선, sagittal slice들을 좌->우로 쌓아 3D volume을 생성하였습니다. 이후, 해당 volume을 axial 축 방향으로 읽어 axial slice를 생성하였습니다.

<img width="990" height="593" alt="Image" src="https://github.com/user-attachments/assets/9a120fca-2311-4ba1-9c5f-2cd3cfff63e8" />

<div align="center"><b>&lt;001_Mild_E의 sag->ax 변환 &gt;</b></div>
<br>

<img width="906" height="593" alt="image" src="https://github.com/user-attachments/assets/93ed7172-f89a-4c84-bd6b-e88f1f437c2e" />
<div align="center"><b>&lt;005_Mod_H의 sag->ax 변환 &gt;</b></div>
<br>
위에서 확인할 수 있듯이, 변환 시 사진의 좌우 scale이 바뀌는 현상이 발생합니다. 이는 sag로 volume 생성 -> ax로 변환 시 출력되는 이미지는 pixelspace가 (row간격, column 간격)=(기존 pixelspacing,기존 slicespacing)이 되기 때문입니다.
위에서 001_Mild_E는 변환 사진이 실제 axial slice와 scale이 비슷하고, 005_Mod_H는 좌우 scale이 크게 차이나는 이유는
<br>

기존의(pixelspacing, slicespacing) = 변환된 사진의 pixelspace(row간격, column 간격) 이
- 001_Mild_E: (0.49,0.5)
- 005_Mod_H : (0.49, NaN)
<br>
에서 메타데이터에 포함되지 않은 005_Mod_H의 slicespacing값이 0.49보다 높을것이기 때문입니다. (slicethickness는 슬라이스의 관찰 두께이므로 슬라이스 간 간격으로 해석하면 안됨.실제로 001_Mild_E에서 slicethickness는 1,slicespacing값은 0.5임). 따라서,

- SliceSpacing 데이터가 존재하는 경우 -> 기존 pixelspacing과 10% 이내로 차이나면 그대로 사용(차이가 크면 아래 방법)
- SliceSpacing 데이터가 존재하지 않는 경우 -> study들을 관찰한 결과 보통 512x312의 비율을 가짐 -> resize(interpolation)

 <br>
이후, 실제 scale을 맞췄다면 아직 직사각형인 사이즈를 대부분의 정사각형 사이즈로 맞추기위해 좌우 빈 공간을 검은 배경으로 패딩하는 알고리즘을 통해 모든 데이터를 axial slice로 관찰하였습니다.

최종 변환된 각 study의 axial slice는 다음과 같습니다.

<img width="1012" height="593" alt="image" src="https://github.com/user-attachments/assets/3f914de1-87ec-4b6a-a8da-15b8f6b70cbf" />
<br>
<div align="center"><b>&lt;scale 차이 문제를 해결&gt;</b></div>

<img width="2449" height="985" alt="image" src="https://github.com/user-attachments/assets/fe7ac7a1-100e-4eb7-ac10-feabb0a74dbc" />

<div align="center"><b>&lt;scale 차이 문제를 해결+배경 패딩을 통한 동일 사이즈화(10개 study의 대표사진)&gt;</b></div>
<br>



---

모델링 파트에선 메모리와 데이터 양의 한계를 고려해, 2.5D 모델로 슬라이스 별 예측을 수행한 뒤 예측 결과를 study 단위의 3D volume [slice, row, column]으로 합산하는 방식을 택했습니다. 
> 본 과제에서 모델 성능은 상관없지만, 저는 3단계 모델을 거치는 구조를 고안했기에 일단 모델 학습이 되어야 뒷 단계로 넘어갈수 있었습니다. 따라서 실질적으로 주어진 리소스로 3개 모두 학습이 가능한 2.5D로 시도하였습니다.
2.5D 모델의 경량성과 3D 구조 기반의 후처리를 동시에 적용하고자 했습니다.

2D 모델링을 할 경우 edema의 slice 별 연결을 학습하지 못한다는 단점이 있습니다. 따라서, [앞 슬라이스, 현재 슬라이스, 뒤 슬라이스]를 3채널로 묶어 입력으로 사용하고, 출력은 현재 슬라이스 기준으로 내놓는 2.5D 방식을 고안했습니다. 
<br>
이를 통해 인접 슬라이스의 맥락을 반영하여 병변이 얇게 걸쳐있는 슬라이스도 앞뒤 정보를 바탕으로 더 잘 감지할 수 있을 것으로 생각했니다.  
(양방향 lstm을 통해 슬라이스 관계를 hidden state로 넣는 것도 생각해보았지만, 이러면 기대하고 있던 시간적 우위를 놓칠 수 있다는 판단을 했습니다.)
 

> 실제 업무에서 리소스와 정확한 mask가 제공된 상황이었다면 동일 모델 구조를 3D에 적용하거나, 후처리를 upgrade하여 2.5D 모델의 경량성을 가져가는 방향을 생각해 볼 것 같습니다.

## 모델 구조와 학습 방식

모델은 3단계 구조를 거칩니다.

### 1단계: CNN Classifier (유효 슬라이스 판별)

해당 사진이 edema 여부를 논할 만한 가치가 있는 유효한 사진인지 검토합니다. MRI 슬라이스의 양끝단에서 나타나는 형체를 알 수 없는 이미지 혹은 두개골만 나타난 이미지를 걸러내는 역할을 합니다.

> 간단한 판별이기에 복잡한 모델이 필요 없다고 생각하여 단순 CNN을 선택했습니다.

**학습 데이터**
- X: `flair_volumes_axial_all` 슬라이스 (2.5D 3채널, 정규화)
- Y: 전처리 기준으로 레이블링된 데이터 (유효=1, 비유효=0)

전처리 기준은 두 가지입니다.
1. `std <= 35`인 슬라이스 제거 (전체적으로 뿌연 이미지)
2. 상하좌우 각 방향의 중간 지점(한 줄)에서 탐색을 시작하여 처음으로 `thresh(=30)` 이상의 픽셀을 만나는 위치가 이미지 크기의 `margin_ratio(=0.3)` 이내에 있어야 유효 (상하좌우 중 두 군데 이상에서 만족해야 함)

<div align="center">
  <img width="174" height="176" alt="Image" src="https://github.com/user-attachments/assets/552a1a7e-65b4-49fa-93b9-f40b0a7b0a42" />
  <img width="174" height="176" alt="Image" src="https://github.com/user-attachments/assets/38ae5f69-15f0-4718-ace5-25b65d6cbcb5" />
  <br>
  <<b>좌</b> : std가 35 이하인 뿌연 이미지, <b>우</b> : std는 35보다 높지만, 끝단 데이터라 가운데 두개골만 포함된 이미지>
</div>

   
<br>
즉, 모델 1은 inference 시 입력되는 데이터를 자동 전처리해주는 효과가 있습니다.  
(edema 논의와 무관한 사진은 masking 작업 없이 종료)

모델은 inference 시 결과로 **유효 확률(0~1)** 을 내놓습니다.

---

### 2단계: ResNet18 (Edema 유무 판별)

Edema 논의가 이루어지는 슬라이스에서, 실제 edema가 존재하는지 유무를 판단합니다.

> 1단계보다 복잡한 판단이라 생각하여 이미지 분류의 표준 모델인 ResNet18을 선택했습니다. Pretrained weight를 활용한 transfer learning을 통해 적은 데이터 양과 부족한 모델 학습 시간을 보완하고자 했습니다.

**학습 데이터**
- X: `flair_volumes_axial_all` 슬라이스 (Sagittal→Axial 변환 포함, 2.5D 3채널)
- Y:
  - Edema 있음 (1): `Mild_E`, `MOD_E`, `modESevH`
  - Edema 없음 (0): `Baseline`, `Mod_H`, `Sev_H`

주어진 데이터 폴더명에서, `Mild_E`는 경증 edema 등으로 해석하여 위와 같이 레이블링하였습니다.

모델은 inference 시 결과로 **edema 확률(0~1)** 을 내놓습니다.

---

### 3단계: U-Net (병변 위치 찾기)

Edema가 존재한다고 판단된 슬라이스 내부에서, 실제 edema 위치를 찾는 역할을 합니다.

> 의료 영상 segmentation 표준 모델로 알려져 있으며, skip connection으로 위치 정보를 보존하기에 적합한 u-net을 선택했습니다. 데이터가 적은 과제 상황에도 맞다고 생각했습니다.

**학습 데이터**
- X: `flair_volumes_axial_all` 슬라이스 (2.5D 3채널, 정규화)
- Y: Border 제거 후 상위 1% 밝기 픽셀 mask (학습용 reference mask)

**학습용 reference mask 생성 방식**

단순히 상위 1% 밝기 픽셀로 mask를 잡는 경우 두개골이 지속적으로 잡히는 문제가 있었습니다.
<br><br>

<img width="1658" height="590" alt="Image" src="https://github.com/user-attachments/assets/dd5dad02-afea-4ede-9975-b0fa85dfad4b" />  

<img width="1658" height="590" alt="Image" src="https://github.com/user-attachments/assets/0767b279-e947-4d2c-bdfb-c2fab7ea7c95" />

  <div align="center"><b>&lt;border 제거 이전&gt;</b></div>

<br>

따라서 4방향(상하좌우)에서 각 행/열을 순회하며 처음으로 `thresh(=30)` 이상의 픽셀을 만나는 위치부터 `band_width(=20px)` 범위를 0으로 마스킹한 뒤, 남은 부분 중 상위 1% 밝기를 가지는 부분을 mask로 지정했습니다.  
<br>

<img width="1658" height="590" alt="Image" src="https://github.com/user-attachments/assets/4366fcd0-2626-489d-b50a-d966955f55f4" />

<img width="1658" height="590" alt="Image" src="https://github.com/user-attachments/assets/02b2ef66-6ae8-446c-9722-6d04fd3bb60e" />
<div align="center"><b>&lt;border 제거 이후&gt;</b></div>
<br><br>

**Loss: Dice Loss**

Edema 영역은 전체 픽셀의 약 1%에 불과한 극심한 클래스 불균형이 존재합니다. BCE Loss만 사용하면 모델이 전부 0으로 예측해도 낮은 loss를 기록하는 문제가 발생하므로, 예측 mask와 정답 mask의 겹치는 비율을 직접 최적화하는 Dice Loss를 선택했습니다.

모델은 inference 시 결과로 **픽셀별 edema 확률값**을 내놓으며, 이를 기반으로 mask를 생성합니다.

---

## Inference 및 후처리 방식


### Inference

```
DICOM 데이터
        ↓
방향 판별 (ImageOrientationPatient)
Sagittal → Axial 변환 (transpose + resize)
        ↓
flair_volumes_axial_all 통합
        ↓
Min-max 정규화
        ↓
256x256 Resize (bilinear interpolation)
        ↓
1단계: CNN Classifier(유효한가?) (confidence_thresh=0.4) [3채널(학습시간이 부족해 일단1채널을 사용했습니다)]
        ↓ 통과          └─ 미통과 ──────────────────→ mask 없음 (종료)
2단계: ResNet18(edema가 있는가?) (edema_thresh=0.2) [3채널 2.5D]
        ↓ 통과          └─ 미통과 ──────────────────→ mask 없음 (종료)
3단계: U-Net 예측(edema가 어디있는가?) → 확률값 (0~1) [3채널 2.5D]
        ↓
원본 크기로 복원 (bilinear interpolation)
        ↓
0.5 Threshold → 이진 mask
```

> *thresh 선택 기준 : 두 모델 모두 학습 데이터가 적고 신뢰도가 낮아서, 기준을 너무 높게 잡으면 실제 edema가 있는 케이스도 걸러버릴 수 있기에 보수적으로 접근했습니다. 1단계의 경우 비교적 학습이 잘 되어 0.5에서 소폭 하향 했고, 2단계의 경우 클래스 불균형으로 학습이 불안정하기에 일단 3단계로 보내는게 나을것이라 판단해 0.2의 값을 주었습니다.

🔴 중간에 통과하지 못하는 슬라이스들은 mask 없이 종료됩니다.  
<br>
<br>

-2단계 모델에 의해 기존에 edima가 없다고 가정했던 7개 study는 모두 mask 없이 나오는 상황입니다.

### 후처리: 3D로 합산 및 1mm 이하 Candidate 제거
```
예측 이진 mask (슬라이스별 2D)

↓

study 단위 3D volume으로 합산

[slice, row, column]

↓

scipy.ndimage.label()

: 슬라이스 간 연결까지 고려하여 3D connected component별로 번호 부여

↓

각 candidate 정보 계산

: voxel count, bounding box, 최대 길이(mm)

slice 방향: (bbox 길이) × slice_spacing
row/col 방향: (bbox 길이) × pixel_spacing

↓

최대 길이 1mm 이하 candidate 제거

↓

후처리 완료 3D mask

[slice, row, column]
```


`pixel_spacing`은 사진별로 픽셀이 의미하는 실제 길이가 다르기 때문에 각자 길이 환산을 해주기 위해 사용했습니다.

`slice_spacing`은 값이 비어있는경우도 많기에 study별, 촬영 방향별로 아래 기준으로 계산하였습니다.


**Sagittal → Axial 변환 study**

- `SpacingBetweenSlices` 태그가 없거나 차이가 큰 경우 (512x312 resize 적용 : 약 5:3비율):
  - `slice_spacing = pixel_spacing × (512 / TARGET_RATIO) / 원본_슬라이스_수` 로 역산
  - 즉, 통상적으로 변환 시 axial 이미지가 가지는 5:3 비율을 나타내도록 하는 slice_spacing을 역산하여 사용하였음.

**Axial study**
- `SpacingBetweenSlices` 태그가 존재하면 그대로 사용
- 없는 경우 `SliceThickness` 사용(이것도 위의 역산방식을 적용하면 좋지만, 변환 시 sag이미지가 가지는 통상적인 비율을 생각하기 어려움)
---
## 1mm 이하 Candidate 제외 기준

candidate가 가지는 bounding box의 모서리 길이 중 최댓값을 해당 candidate의 최대 길이라고 가정하였습니다. 
실제 최대 길이는 오차가 있겠지만, 제품의 시간과 작은 thresh를 생각했을때 이정도로 충분하다고 생각했습니다.

🔴 아래부터 등장하는 사진은 2D 모델링 기준 결과입니다. 이해를 돕기 위한 사진이며, 개선한 2.5D 모델에선 모델 학습 시간 및 리소스가 부족해 결과 사진을 뽑아내지 못한 상황입니다.

---


<img width="2345" height="593" alt="Image" src="https://github.com/user-attachments/assets/e1e38b29-1ae5-48e2-b473-137a2d6a9c13" /> 

<div align="center"><b>&lt;후처리 전,후 predict mask(3D) 단면 &gt;</b></div>
<br>
후처리 이후, 부피가 가장 큰(voxel_count가 가장 높은) candidate를 시각화해보면 다음과 같습니다.  


<img width="1746" height="593" alt="Image" src="https://github.com/user-attachments/assets/38efd85c-680f-4193-ad85-6739464f892a" /> 

<div align="center"><b>&lt;largest candidate를 가장 많이 포함하는 slice&gt;</b></div>

---

**최종 inference 결과(3D mask) overlay를 축 방향 gif로 표현하면 다음과 같습니다.**
<br>
> 애초에 edima가 없다고 분류하여 학습시킨 7개의 폴더에 대해선 대부분 mask가 생성되지 않았습니다. 즉, 2단계 모델에 의해 최종 result엔 mask가 없는 것이 대부분입니다.

---

<br>
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/96f15f22-bc73-4cab-a4f0-cf27e45e82b9" />

---

<br>
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/d713bcb7-0f6d-4f79-86f9-89e5f9bc5588" />

---

<br>
<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/505a9730-236a-4523-b090-0d0adbdbc057" />

---

## 모델 성능 평가 방법

충분한 양과 질의 데이터가 주어진다면 아래 지표를 통해 평가할 것 같습니다.

- **Dice Coefficient**: 예측 mask와 정답 mask의 겹치는 비율
- **IoU**: 두 mask의 교집합/합집합 비율
- **Sensitivity / Specificity**: 병변을 얼마나 놓치지 않고 잡는지 / 정상을 얼마나 정확히 구분하는지
- **Hausdorff Distance**: 예측 경계와 정답 경계 사이의 최대 거리
- **+mri 병변 데이터에선 빠르게 결과를 도출해내야 하므로, 속도를 기본적인 평가지표로 둘 것 같습니다.**

---

## 구현하면서 둔 가정 및 한계점

- **Reference mask 생성**: 의사 annotation 대신 "border 제거 후 상위 1% 밝기 픽셀 = edema 후보"로 가정했습니다. 또한 Border 제거 시 band_width를 고정값(20px)으로 설정했는데, 해상도가 다른 이미지에서는 적절하지 않을 수 있습니다.

- **2.5D 모델링**: 인접 슬라이스를 3채널로 묶어 앞뒤 맥락을 반영하였으나, 완전한 3D 공간 학습은 아닙니다. 슬라이스 간 공간적 관계를 명시적으로 학습하려면 3D 모델이 더 적합합니다. 하나의 candidate에서 중간 슬라이스가 걸러지면 여러 개의 candidate로 분리되는 오류가 발생할 수 있으나, 2.5D를 통해 인접 슬라이스 맥락을 반영하여 이를 어느정도 완화하고자 했습니다.
> 즉, 1,2단계 모델에서 구현된 유효하지 않은/ edema가 없다고 판단되는 slice(단면) out의 알고리즘은 사실 3D에선 특정 크기의 box로 (공간)을 훑으며 공간을 out시키는 알고리즘으로 가야할 듯 합니다.
> 다만 현재는 데이터/리소스의 한계, 3D에서 학습용 reference mask 생성의 어려움 으로 2.5D로 진행하였습니다.

- **Sagittal → Axial 변환**: Axial과 Sagittal은 동일한 3D 공간을 다른 축으로 슬라이싱한 것이므로 단순 축 변환(transpose)만으로 방향 통일이 가능합니다. 다만 `SpacingBetweenSlices`가 없거나 `PixelSpacing`과 차이가 큰 경우 512x312로 resize하였는데, 이 비율을 임의로 설정하였습니다.

- **slice_spacing 역산**: Sagittal → Axial 변환 후 512x312 resize가 적용된 경우, 후처리에서 slice 방향 실제 간격을 `pixel_spacing × (512 / TARGET_RATIO) / 원본_슬라이스_수`로 역산하여 사용하였습니다. 실제 SliceSpacing과 오차가 있을 수 있습니다.


- **입력 크기 통일**: study마다 이미지 크기가 다르고, 학습의 경량을 위해 256x256으로 resize하여 학습했습니다. 이 과정에서 정보 손실 가능성이 있습니다.

- **Edema 레이블**: study 폴더 이름을 기반으로 수동 레이블링을 적용했습니다. `Mod_H`, `Sev_H`는 Hemorrhage만 있는 것으로 해석해 Edema 없음(0)으로 처리했습니다. 이 과정에서 Edema 3개 vs 비Edema 7개 study에서 발생하는 클래스 불균형이 있었고, `pos_weight`로 보정하였습니다.

- **데이터의 한계**: 데이터의 양이 부족하며, 정확한 mask가 없었습니다. 이에 전처리 과정에서의 thresh도 어느정도 실험을 해가며 수동으로 조정하는 수 밖에 없었습니다.
- **모델 구조의 한계**: 3단계 모델 구조에서 앞 단계의 오류가 뒷 단계로 전파됩니다. 1단계에서 잘못 걸러진 슬라이스는 2, 3단계에서 복구할 수 없습니다. 또한 현재는 1단계 CNN Classifier의 학습 정답(label)이 규칙 기반 전처리 결과이므로, 전처리 로직의 한계가 그대로 모델에 반영됩니다.
- **하드웨어** : 부족한 gpu 리소스와 cpu만으로 돌리니 충분한 학습과 파라미터 튜닝이 이루어지지 못했습니다.(성능 평가가 목적은 아니므로, 에포크를 매우 낮게 설정했습니다.)

---
 

## 실제 업무라면 추가로 확인하거나 질문하고 싶은 사항

- **2D + 후처리 가능성**: 시간을 생각했을때 2D 모델이 3D보다 우위에 있다고 생각합니다. 2D모델에서 이루어지는 slice 간 edema 정보의 손실을 후처리 알고리즘을 통해 어느정도 해결할 수 있을지 궁금합니다. 이번 과제에서도 매우 작은 edema 제거는 후처리에 의해 3D와 유사하게 적용되었다고 생각합니다. 위에서 언급했던 (하나의 candidate가 여러개의 candidate로 잡힐 수 있다는 문제) 를 포함한 여러 문제점과 해결 방안이 궁금합니다.

- **2.5D vs 3D 실제 성능 차이**: 데이터가 충분한 환경에서 2.5D와 3D 모델의 실제 성능 차이가 얼마나 되는지 궁금합니다. 2.5D가 메모리, 속도, pretrained 활용 측면에서 유리하지만, 슬라이스 간 공간적 관계를 명시적으로 학습하지 못하는 한계가 실제 임상에서 어느 정도 영향을 미치는지 궁금합니다.

- **Annotation 기준**: ARIA-E로 판독하는 정확한 임상 기준, WMH와 ARIA-E 구분 방법이 궁금합니다. 손수 masking된 데이터를 많이 확보하기 어려울 것 같은데, 로직에 의해 구현되는지 궁금합니다. 동일 환자의 Baseline과 ARIA-E 데이터를 쌍으로 활용하는 방식이 유효한지도 궁금합니다.

- **재학습**: 의사가 예측 결과를 검토/수정할 수 있는 피드백 루프가 있는지, 있다면 그 데이터를 재학습에 활용하는지 궁금합니다.

- **1단계 모델의 실질적 목표**: '양끝단 비유효 이미지(공간) 필터링'이 실제로 많은 데이터가 주어진 환경에서는 2단계 혹은 3단계 모델 학습에 포함될 수 있을 것 같은데, 시간 비용을 줄이기 위해 이를 하나의 모델에 포함해 학습시키는지, 혹은 독립적인 전처리 과정으로 가지는지 궁금합니다.

- **단일 모델로 병변 유무 + 위치 동시 처리**: 특정 영역을 '있다면' 표시해주는 방법이 하나의 모델링으로 가능할지 궁금합니다. 방대한 데이터가 있다면 voxel별 확률 학습 후 threshold를 높게 잡으면 가능할 것 같은데, 일반적으로 잘 이루어지지 않는 것 같습니다.

- **모델 구조의 개선**: 현재 3단계 파이프라인은 앞 단계에서 걸러진 슬라이스가 이후 단계에서 복구되지 않는 구조입니다. U-Net의 skip connection과 비슷한 외형을 통해 낮은 확률로 걸러진 슬라이스를 재검토하는 구조가 성능 개선에 유효할지(시간비용 대비) 궁금합니다.

- **slice_spacing 역산의 타당성**: Sagittal → Axial 변환 후 SliceSpacing이 없는 경우 5:3 비율 기준으로 역산하여 사용하였는데, 실제 임상 환경에서 이 근사값이 후처리 결과에 미치는 영향이 어느 정도인지 궁금합니다. 또한 정확한 SliceSpacing을 확보하는 방법이 있는지도 궁금합니다.
