# FLAIR 기반 ARIA-E Segmentation Pipeline

코랩에서 작성하였습니다. ipynb 파일 용량이 높아지며 github에 업로드하지 못하는 문제가 있었어서, 전처리 및 파라미터 튜닝을 위한 시각화 과정은 대부분 주석처리 해놓은 상태입니다. 
모두 돌아가는데 그대로 돌릴 경우 3~4분, 주석을 모두 풀고 돌릴 경우 10분정도 걸립니다.(모델학습 파트 제외)

## 실행 방법

해당 코드 및 기존 데이터 폴더와 첨부된 `checkpoint_classifier.pth`, `checkpoint_edema_classifier.pth`, `checkpoint.pth`을 구글 드라이브에 업로드한 상태에서 그대로 모두 실행하시면 됩니다 (코랩 환경 기준, 파일 저장 경로 : /content/drive/MyDrive).

> 모델 학습 시간이 오래 걸려 결과만 저장한 채로 주석 처리해놓았습니다. 구글 드라이브에서 저장된 모델을 불러오게 되어있으니 주석 제거 없이 실행하시면 됩니다.

조금 더 시간을 단축하기 위해선, 첨부된 `dicoms_preprocessed`, `results` 폴더도 구글 드라이브에 업로드 후 실행하시면 좋습니다. (업로드하지 않고 실행하면 알아서 새로 폴더를 생성합니다.)

https://drive.google.com/drive/folders/1tG9SSYavzfjQ0-b7x612tnhR-548haAO?usp=sharing
(구글드라이브 링크에 모두 업로드되어있습니다. 공유폴더로 지정하시면 업로드 생략 가능할 것 같습니다.)

### 파이프라인 순서

1. 데이터 탐색
2. 전처리
3. Reference mask 생성
4. 모델 정의 (CNN Classifier / EDEMA Classifier / U-Net)
5. 각 모델 학습 또는 checkpoint 로드
6. Inference (3단계 파이프라인)
7. 후처리
8. 결과 저장 및 시각화

---

## 모델 구조와 학습 방식

모델은 3단계 구조를 거칩니다.

### 1단계: CNN Classifier (유효 슬라이스 판별)

해당 사진이 edema 여부를 논할 만한 가치가 있는 유효한 사진인지 검토합니다. MRI 슬라이스의 양끝단에서 나타나는 형체를 알 수 없는 이미지 혹은 두개골만 나타난 이미지를 걸러내는 역할을 합니다.

간단한 판별이기에 복잡한 모델이 필요 없다고 생각하여 단순 CNN을 선택했습니다.

**학습 데이터**
- X: 주어진 FLAIR 슬라이스 전체 (정규화)
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

   

즉, 모델 1은 inference 시 입력되는 데이터를 자동 전처리해주는 효과가 있습니다.  
(edema 논의와 무관한 사진은 masking 작업 없이 종료)

모델은 inference 시 결과로 **유효 확률(0~1)** 을 내놓습니다.

---

### 2단계: ResNet18 (Edema 유무 판별)

Edema 논의가 이루어지는 슬라이스에서, 실제 edema가 존재하는지 유무를 판단합니다.

1단계보다 복잡한 판단이라 생각하여 이미지 분류의 표준 모델인 ResNet18을 선택했습니다. Pretrained weight를 활용한 transfer learning을 통해 적은 데이터 양과 부족한 모델 학습 시간을 보완하고자 했습니다.

**학습 데이터**
- X: `flair_paths_preprocessed` (전처리된_유효한 슬라이스)
- Y:
  - Edema 있음 (1): `Mild_E`, `MOD_E`, `modESevH`
  - Edema 없음 (0): `Baseline`, `Mod_H`, `Sev_H`

주어진 데이터 폴더명에서, `Mild_E`는 경증 edema 등으로 해석하여 위와 같이 레이블링하였습니다.

모델은 inference 시 결과로 **edema 확률(0~1)** 을 내놓습니다.

---

### 3단계: U-Net (병변 위치 찾기)

Edema가 존재한다고 판단된 슬라이스 내부에서, 실제 edema 위치를 찾는 역할을 합니다.

의료 영상 segmentation 표준 모델로 알려져 있으며, skip connection으로 위치 정보를 보존하기에 적합한 u-net을 선택했습니다. 데이터가 적은 과제 상황에도 맞다고 생각했습니다.

**학습 데이터**
- X: `flair_paths_preprocessed` (전처리된 슬라이스, 정규화)
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
원본 FLAIR 슬라이스 (전처리 없이 원본 그대로 입력)
        ↓
Min-max 정규화
        ↓
256x256 Resize (bilinear interpolation)
        ↓
1단계: CNN Classifier (confidence_thresh=0.4) 
        ↓ 통과
2단계: ResNet18 (edema_thresh=0.2)
        ↓ 통과
3단계: U-Net 예측 → 확률값 (0~1)
        ↓
원본 크기로 복원 (bilinear interpolation)
        ↓
0.5 Threshold → 이진 mask
```

*thresh 선택 기준 : 두 모델 모두 학습 데이터가 적고 신뢰도가 낮아서, 기준을 너무 높게 잡으면 실제 edema가 있는 케이스도 걸러버릴 수 있기에 보수적으로 접근했습니다. 1단계의 경우 비교적 학습이 잘 되어 0.5에서 소폭 하향 했고, 2단계의 경우 클래스 불균형으로 학습이 불안정하기에 일단 3단계로 보내는게 나을것이라 판단해 0.2의 값을 주었습니다.

중간에 통과하지 못하는 슬라이스들은 mask 없이 종료됩니다.

### 후처리: 1mm² 이하 Candidate 제거

```
예측 이진 mask
        ↓
scipy.ndimage.label()
: 연결된 픽셀 덩어리(connected component)별로 번호 부여
        ↓
각 덩어리 면적 계산
: component_size(px) × pixel_spacing² → mm² 단위 환산
        ↓
1mm² 이하 덩어리 제거
        ↓
후처리 완료 mask
```

`pixel_spacing`은 사진별로 픽셀이 의미하는 실제 길이가 다르기 때문에 각자 길이 환산을 해주기 위해 사용했습니다.

<img width="2345" height="593" alt="Image" src="https://github.com/user-attachments/assets/46a21c05-1825-42e9-8f39-5bdcf63949f4" />
<div align="center"><b>&lt;원본, predict mask, 후처리된 predict mask, 최종 overlay &gt;</b></div>

---

**최종 inference 결과는 다음과 같습니다.**
<br>
(애초에 edima가 없다고 분류하여 학습시킨 7개의 폴더에 대해선 대부분 mask가 생성되지 않았습니다. 즉, 2단계 모델에 의해 최종 result엔 mask가 없는 것이 대부분입니다.)

---

<br>
<img width="1746" height="593" alt="Image" src="https://github.com/user-attachments/assets/d67a6dbc-3a61-4ae2-a0eb-190e135068d3" />
<div align="center"><b>&lt;최종 출력 case 1&gt;</b></div>

---

<img width="2958" height="593" alt="Image" src="https://github.com/user-attachments/assets/00059f13-241d-41f2-95dc-ddac3e733877" />
<div align="center">원본 &nbsp;|&nbsp; 1% 밝기 &nbsp;|&nbsp; border 제거 후 1% 밝기 (정답지) &nbsp;|&nbsp; predict &nbsp;|&nbsp; 후처리된 최종결과</div>
<div align="center"><b>&lt;최종 출력 case 2 : 1단계 모델에서 out(유효하지 않은 사진)되어 최종 mask X(제일 오른쪽 사진)&gt;</b></div>

---

<img width="2944" height="593" alt="Image" src="https://github.com/user-attachments/assets/7ce8fbb9-4445-4b8b-8dea-fa2fc7f9762a" />
<div align="center"><b>&lt;최종 출력 case 3 : 2단계 모델에서 out(edema 영역 없다고 판단)되어 최종 mask X(제일 오른쪽 사진)&gt;</b></div>

---
<img width="2944" height="593" alt="Image" src="https://github.com/user-attachments/assets/aa207b4e-11dd-4717-ae99-64cff538b79d" />
<div align="center"><b>&lt;최종 출력 case 4 - 모든 단계 pass(유효하며 edema 존재), 후처리 이후 최종 mask(제일 오른쪽 사진)&gt;</b></div>



---

## 1mm 이하 Candidate 제외 기준

공간 상의 edema이기에 실제로는 부피를 기준으로 제외하는 게 맞다고 생각했습니다. 하지만 단면 슬라이스를 보유하고 있으므로 **넓이(면적)** 를 기준으로 제거하였습니다.

---

## 모델 성능 평가 방법

충분한 양과 질의 데이터가 주어진다면 아래 지표를 통해 평가할 것 같습니다.

- **Dice Coefficient**: 예측 mask와 정답 mask의 겹치는 비율
- **IoU**: 두 mask의 교집합/합집합 비율
- **Sensitivity / Specificity**: 병변을 얼마나 놓치지 않고 잡는지 / 정상을 얼마나 정확히 구분하는지
- **Hausdorff Distance**: 예측 경계와 정답 경계 사이의 최대 거리

---

## 구현하면서 둔 가정 및 한계점

- **Reference mask 생성**: 의사 annotation 대신 "border 제거 후 상위 1% 밝기 픽셀 = edema 후보"로 가정했습니다. 또한 Border 제거 시 band_width를 고정값(20px)으로 설정했는데, 해상도가 다른 이미지에서는 적절하지 않을 수 있습니다.
- **2D slice 단위 처리**: 3D 볼륨 전체가 아닌 슬라이스별 독립 처리로, 슬라이스 간 연속성을 고려하지 못했습니다.
- **입력 크기 통일**: study마다 이미지 크기가 달라 256x256으로 resize하여 학습했습니다. 이 과정에서 정보 손실 가능성이 있습니다.
- **Edema 레이블**: study 폴더 이름을 기반으로 수동 레이블링을 적용했습니다. `Mod_H`, `Sev_H`는 Hemorrhage만 있는 것으로 해석해 Edema 없음(0)으로 처리했습니다. 이 과정에서 Edema 3개 vs 비Edema 7개 study에서 발생하는 클래스 불균형이 있었고, `pos_weight`로 보정하였습니다.
- **검증셋 생략**: 데이터가 10개 study로 매우 적어 의미있는 검증셋 분리가 불가능하여 생략하였습니다. 모든 데이터를 학습에 활용하였으며, 이 때문에 2단계 모델은 과적합된 경향을 보이는 것 같습니다.
- **데이터의 한계**: 데이터의 양이 부족하며, 정확한 mask가 없었습니다. 이에 전처리 과정에서의 thresh도 어느정도 실험을 해가며 수동으로 조정하는 수 밖에 없었습니다.
- **모델 구조의 한계**: 3단계 모델 구조에서 앞 단계의 오류가 뒷 단계로 전파됩니다. 1단계에서 잘못 걸러진 슬라이스는 2, 3단계에서 복구할 수 없습니다. 또한 현재는 1단계 CNN Classifier의 학습 정답(label)이 규칙 기반 전처리 결과이므로, 전처리 로직의 한계가 그대로 모델에 반영됩니다.
- **하드웨어** : 부족한 gpu 리소스와 cpu만으로 돌리니 충분한 학습과 파라미터 튜닝이 이루어지지 못했습니다.(성능 평가가 목적은 아니므로, 에포크를 매우 낮게 설정했습니다.)

---

## 실제 업무라면 추가로 확인하거나 질문하고 싶은 사항

- **Annotation 기준**: ARIA-E로 판독하는 정확한 임상 기준, WMH와 ARIA-E 구분 방법이 궁금합니다. 손수 masking된 데이터를 많이 확보하기 어려울 것 같은데, 로직에 의해 구현되는지 궁금합니다. 동일 환자의 Baseline과 ARIA-E 데이터를 쌍으로 활용하는 방식이 유효한지도 궁금합니다.
- **재학습**:의사가 예측 결과를 검토/수정할 수 있는 피드백 루프가 있는지, 있다면 그 데이터를 재학습에 활용하는지 궁금합니다.
- **후처리**: 슬라이스 단위 2D 처리와 3D 볼륨 처리 중 어떤 것이 임상적 근거에 가까운지, 비용과 시간의 측면에서 어떤 선택이 정답에 가까울지 궁금합니다.
- **1단계 모델의 실질적 목표**: '양끝단 비유효 이미지 필터링'이 실제로 많은 데이터가 주어진 환경에서는 2단계 혹은 3단계 모델 학습에 포함될 수 있을 것 같은데, 시간 비용을 줄이기 위해 이를 하나의 모델에 포함해 학습시키는지, 혹은 독립적인 전처리 과정으로 가지는지 궁금합니다.
- **단일 모델로 병변 유무 + 위치 동시 처리**: 특정 영역을 '있다면' 표시해주는 방법이 하나의 모델링으로 가능할지 궁금합니다. 방대한 데이터가 있다면 픽셀별 확률 학습 후 threshold를 높게 잡으면 가능할 것 같은데, 일반적으로 잘 이루어지지 않는 것 같습니다.
- **모델 구조의 개선** 현재 3단계 파이프라인은 앞 단계에서 걸러진 슬라이스가 이후 단계에서 복구되지 않는 구조입니다. U-Net의 skip connection과 비슷한 외형을 통해 낮은 확률로 걸러진 슬라이스를 재검토하는 구조가 성능 개선에 유효할지(시간비용 대비) 궁금합니다.
