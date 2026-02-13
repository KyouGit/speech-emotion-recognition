# Korean Speech Emotion Recognition

한국어 음성 감정 인식 시스템. 다양한 음성 특징 추출 방법과 멀티모달 접근법을 비교 실험한 프로젝트.

## Dataset

**감정 분류를 위한 대화 음성 데이터셋** (AI Hub)

| 항목 | 내용 |
|------|------|
| 총 샘플 수 | 43,975개 wav 파일 |
| 화자 수 | 59명 |
| 감정 클래스 | 7개 (angry, disgust, fear, happiness, neutral, sadness, surprise) |
| 데이터 분할 | Speaker-Independent Split (Train 30명 / Val 13명 / Test 16명) |

```
data/emotions/
├── angry/        (2,481 test samples)
├── disgust/      (931)
├── fear/         (845)
├── happiness/    (753)
├── neutral/      (550)
├── sadness/      (2,992)
└── surprise/     (274)
```

## Experiments

### Exp 1. MFCC + CNN (Baseline)

MFCC 40차원 spectrogram을 CNN으로 학습하는 전통적 방식.

```
Audio → MFCC (40-dim spectrogram) → CNN → 7 emotions
```

- **Test Accuracy: 74.98%**
- 단순하지만 가장 강력한 baseline

### Exp 2. Wav2Vec2 Frozen

Facebook의 `wav2vec2-base-960h` 사전학습 모델을 frozen feature extractor로 사용.

```
Audio → Wav2Vec2 (frozen) → Classifier Head → 7 emotions
```

| Split | Test Accuracy |
|-------|-------------|
| Random | 52.50% |
| Speaker-Independent | 39.33% |

- 영어 음성으로 사전학습된 모델이라 한국어 감정에 부적합
- Random split(52.50%) vs Speaker split(39.33%) 차이 → Random split은 화자 정보 누출로 과대평가

### Exp 3. Wav2Vec2 Fine-Tune

Wav2Vec2의 마지막 4개 transformer layer를 unfreeze하여 fine-tuning.

```
Audio → Wav2Vec2 (last 4 layers unfrozen) → Classifier Head → 7 emotions
```

- **Test Accuracy: 63.24%** (Best Val: 63.51%)
- Frozen(39.33%) 대비 +24%p 개선, 그러나 MFCC baseline(74.98%)에 미달
- 영어 사전학습 모델의 한국어 적용 한계

### Exp 4. Multimodal: MFCC + KoBERT (Whisper STT)

Whisper로 음성을 텍스트로 변환 후, MFCC(음성) + KoBERT(텍스트) late fusion.

```
Audio ──┬── MFCC (40-dim, global avg pool) ──────────────┐
        │                                                 ├── Concat(808) → MLP → 7 emotions
        └── Whisper STT → Text → KoBERT [CLS] (768-dim) ─┘
```

- KoBERT: `kykim/bert-kor-base` (frozen, feature extractor)
- MLP: Linear(808→256) → GELU → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→7)

| Whisper Model | Test Accuracy | Best Val Acc |
|---------------|-------------|-------------|
| small | 72.99% | 75.15% |
| medium | 74.54% | 76.24% |

- Whisper 품질 향상(small→medium)으로 +1.55%p 개선
- KoBERT가 frozen 상태라 감정 특화 표현을 학습하지 못함
- 768차원의 BERT 피처가 40차원 MFCC에 비해 노이즈로 작용할 가능성

## Results Summary

Speaker-Independent Split 기준 전체 결과:

| # | Model | Test Acc | Best Val Acc | vs Baseline |
|---|-------|----------|-------------|-------------|
| 1 | **MFCC + CNN (Baseline)** | **74.98%** | - | - |
| 2 | Wav2Vec2 Frozen | 39.33% | - | -35.65%p |
| 3 | Wav2Vec2 Fine-Tune | 63.24% | 63.51% | -11.74%p |
| 4 | Multimodal (Whisper small) | 72.99% | 75.15% | -1.99%p |
| 5 | Multimodal (Whisper medium) | 74.54% | 76.24% | -0.44%p |

## Key Findings

1. **MFCC가 사전학습 모델보다 우수**: 영어 사전학습 음성 모델(Wav2Vec2)은 한국어 감정 인식에서 MFCC보다 크게 저조
2. **Speaker-Independent 평가의 중요성**: Random split은 화자 정보 누출로 성능이 과대평가됨 (Wav2Vec2: 52.50% → 39.33%)
3. **멀티모달 접근의 가능성과 한계**: Whisper STT + KoBERT 텍스트 피처 추가 시 baseline에 근접하지만, frozen KoBERT의 일반적 텍스트 임베딩은 감정 분류에 최적화되지 않음
4. **STT 품질의 영향**: Whisper small → medium 업그레이드로 전사 품질이 향상되어 +1.55%p 개선 확인

## Usage

### Requirements

```bash
pip install -r requirements.txt
pip install transformers soundfile openai-whisper
```

### Training

```bash
# MFCC baseline (speaker-independent split)
python main.py --model mfcc --split speaker

# Wav2Vec2 frozen
python main.py --model wav2vec2 --split speaker

# Wav2Vec2 fine-tune
python main.py --model wav2vec2ft --split speaker

# Multimodal (MFCC + Whisper STT + KoBERT)
python main.py --model multimodal --split speaker
```

### Output Files

각 실험 실행 시 생성되는 파일:

| File | Description |
|------|-------------|
| `best_model_*.pth` | 최고 성능 모델 체크포인트 |
| `training_history_*_speaker.png` | Train/Val Loss, Accuracy 곡선 |
| `confusion_matrix_*_speaker.png` | 감정별 혼동 행렬 |
| `results_*_speaker.json` | 최종 평가 수치 |
| `transcriptions.json` | Whisper STT 전사 캐시 (multimodal 전용) |

## Architecture Details

### GPU Memory Management (RTX 4060 Ti 8GB)

멀티모달 파이프라인은 순차적 모델 로드/언로드 방식으로 GPU 메모리를 관리:

1. **Whisper** (~1-3GB) → 전사 완료 후 언로드
2. **KoBERT** (~400MB) → [CLS] 추출 완료 후 언로드
3. **MLP Classifier** (~1MB) → 학습

### Speaker-Independent Split

화자 단위로 분할하여 모델이 새로운 화자에 대해 일반화할 수 있는지 평가:

- Train: 30명 (29,466 samples)
- Validation: 13명 (5,683 samples)
- Test: 16명 (8,826 samples)

## Project Structure

```
speech-emotion-recognition/
├── main.py                  # 전체 학습/평가 파이프라인
├── preprocess.py            # 데이터 전처리
├── predict.py               # 추론
├── requirements.txt         # 의존성
├── README.md
├── data/emotions/           # 감정별 wav 파일 (not tracked)
├── results_*.json           # 실험 결과
├── training_history_*.png   # 학습 곡선
├── confusion_matrix_*.png   # 혼동 행렬
└── transcriptions.json      # Whisper STT 캐시 (not tracked)
```

## Tech Stack

- **Framework**: PyTorch, torchaudio
- **Audio Features**: MFCC (40-dim)
- **Speech Model**: Wav2Vec2 (facebook/wav2vec2-base-960h)
- **STT**: OpenAI Whisper (small / medium)
- **Text Model**: KoBERT (kykim/bert-kor-base)
- **Evaluation**: scikit-learn

## Contact

- Email: qsc303@gmail.com
- GitHub: [@KyouGit](https://github.com/KyouGit)

## License

MIT License
