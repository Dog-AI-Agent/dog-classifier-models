# Dog Breed Classifier

Stanford Dogs 데이터셋(120종)을 사용하여 3가지 비전 모델 아키텍처를 **from scratch** 학습하고 성능을 비교하는 프로젝트입니다.

## 모델 아키텍처

| 모델          | timm 식별자                    | 특징                                   |
| ------------- | ------------------------------ | -------------------------------------- |
| ViT-Small     | `vit_small_patch16_224`        | Attention 기반 Vision Transformer      |
| Swin-Tiny     | `swin_tiny_patch4_window7_224` | Shifted Window 기반 계층적 Transformer |
| ConvNeXt-Tiny | `convnext_tiny`                | Transformer 설계를 적용한 Modern CNN   |

## 환경 설정

```bash
conda env create -f environment.yml
conda activate breed-classifier
uv pip install -r requirements.txt
```

**요구사항:** Python 3.11, CUDA 지원 GPU 권장

## 사용법

### 학습

```bash
# 기본 학습 (ViT-Small, 50 epochs)
python train.py --model vit_small

# 하이퍼파라미터 지정
python train.py --model swin_tiny --epochs 100 --batch_size 64 --lr 5e-4 --seed 123

# 체크포인트에서 추가 학습 (30 epoch 더)
python train.py --model vit_small --epochs 30 --resume
```

### 평가

```bash
# 단일 모델 평가
python evaluate.py --model vit_small

# 전체 모델 평가
python evaluate.py --model all
```

### 결과 비교

```bash
python compare.py
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

## 프로젝트 구조

```
breed-classifier/
├── configs/
│   └── default.py          # TrainConfig 데이터클래스 (전체 하이퍼파라미터)
├── src/
│   ├── dataset.py           # Stanford Dogs 다운로드 및 데이터 파이프라인
│   ├── models.py            # timm 모델 팩토리
│   ├── trainer.py           # 학습 루프 (mixed precision, early stopping, resume)
│   ├── metrics.py           # Top-k accuracy, macro F1
│   └── utils.py             # seed 고정, device 감지
├── train.py                 # 학습 진입점
├── evaluate.py              # 테스트셋 평가
└── compare.py               # 모델 간 결과 비교 테이블
```

## 학습 설정

| 항목            | 기본값                    |
| --------------- | ------------------------- |
| Optimizer       | AdamW (weight_decay=0.05) |
| Scheduler       | Cosine Annealing          |
| Label Smoothing | 0.1                       |
| Mixed Precision | 활성화                    |
| Early Stopping  | patience=10               |
| Image Size      | 224x224                   |
| Batch Size      | 32                        |
| Val Split       | 15% (stratified)          |

## 결과

| 모델          | Test Top-1 | Test Top-5 | Test F1 | Best Epoch |
| ------------- | ---------- | ---------- | ------- | ---------- |
| ViT-Small     | -          | -          | -       | -          |
| Swin-Tiny     | -          | -          | -       | -          |
| ConvNeXt-Tiny | -          | -          | -       | -          |

> 학습 완료 후 `python compare.py`로 결과를 확인하고 이 테이블을 업데이트해 주세요.

## 컨벤션

- **설정 변경**: `configs/default.py`의 `TrainConfig` 수정 또는 CLI args로 오버라이드
- **모델 추가**: `TrainConfig.MODEL_REGISTRY`에 timm 모델명 등록 → `train.py`의 `choices`에 추가
- **체크포인트**: `checkpoints/best_{model_name}.pt`에 저장 (model, optimizer, scheduler, scaler, early_stopping 상태 포함)
- **로그**: `logs/{model_name}/`에 TensorBoard 이벤트 저장

## 팀 구성

| 이름            | 역할                        |
| --------------- | --------------------         |
| 김재현          | Project Leader              |
| 최동원          | Optimizing Model            |
| 이민혜          | Model selection             |
| 송진우          | Evaluating Outputs          |
| 장승우          | Test Set Creator            |

---