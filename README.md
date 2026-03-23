# Stanford Dogs 120종 분류 — 백본 비교 실험

Stanford Dogs 데이터셋(120종)을 대상으로 다양한 백본의 전이학습 성능을 비교하는 실험입니다.
Apple M2 Mac 로컬 환경에서 실행됩니다.

## 실험 구성

| Phase | 내용 |
|-------|------|
| Phase 1 | 동일 조건에서 백본만 교체하여 성능·속도·크기 비교 |
| Phase 2 | Phase 1 최고 백본에 proper validation + EarlyStopping 적용 |
| Phase 2 (cont.) | EarlyStopping 미발동 시 이어서 추가 학습 |

## 공통 하이퍼파라미터 (Phase 1)

| 항목 | 값 |
|------|----|
| Input size | 224×224 |
| Batch size | 32 |
| Epochs | 10 |
| LR | 0.0001 |
| Backbone | frozen (trainable=False) |
| Head | GlobalAveragePooling2D → Dense(120, softmax) |

## 비교 백본

- MobileNetV2
- MobileNetV3Large
- EfficientNetB0

## Phase 1 결과

| backbone | val_acc_best | val_top5_best | 추론(ms) | 모델 크기 |
|----------|-------------|---------------|----------|-----------|
| EfficientNetB0 | **0.8552** | **0.9867** | 54.26 | 18.0MB |
| MobileNetV3Large | 0.7818 | 0.9678 | 41.97 | 13.4MB |
| MobileNetV2 | 0.7790 | 0.9672 | 40.11 | 10.9MB |

→ **EfficientNetB0** 가 최고 성능으로 Phase 2 진행

## 핵심 흐름 & 결론

| Phase | 모델 | val_acc | 비고 |
|-------|------|---------|------|
| Phase 1 (baseline) | MobileNetV2 | 77.9% | 기준선 |
| Phase 1 | EfficientNetB0 | 85.5% | 백본 교체만으로 **+7.6%p** |
| Phase 2 | EfficientNetB0 | **86.9%** | EarlyStopping + proper validation 추가 **+1.5%p** |
| Phase 3 | EfficientNetB0 | 86.9% | 추가 학습 시도, 유의미한 개선 없음 → Phase 2 채택 |

**최종 개선폭: MobileNetV2 대비 +9.1%p**

### 요인 분석

- **백본 교체 (+7.6%p)**: 같은 학습 조건에서 MobileNetV2 → EfficientNetB0로만 바꿔도 가장 큰 성능 향상
- **학습 방식 개선 (+1.5%p)**: EarlyStopping + 최적 모델 저장(ModelCheckpoint) + proper validation 분리 적용
- **Phase 3**: 추가 학습을 시도했으나 17 epoch에서 EarlyStopping 발동, 이미 수렴 상태 확인 → Phase 2 모델 최종 채택

### 발표 포인트

> 백본 교체만으로 7%p 이상 올랐고, 거기에 학습 방식(EarlyStopping + 최적 모델 저장)까지 개선해서 총 9%p 향상.
> Phase 3는 "추가 학습을 시도했지만 유의미한 개선 없음 → Phase 2 채택"으로 정리.

## 파일 구조

```
models/
├── backbone_comparison.ipynb   # 실험 노트북
├── requirements.txt
└── experiments/                # 결과 저장 디렉토리
    ├── *.keras                 # 저장된 모델
    ├── *.csv                   # 학습 결과
    └── *.png                   # 학습 곡선 이미지
```

## 실행 방법

```bash
pip install -r requirements.txt
jupyter notebook backbone_comparison.ipynb
```

셀을 순서대로 실행하세요. Phase 1은 백본별 셀을 개별 실행하고, Phase 2는 Phase 1 완료 후 실행합니다.

## 환경

- Apple M2 (Metal GPU)
- TensorFlow 2.16.2
- Dataset: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) via `tensorflow-datasets`

