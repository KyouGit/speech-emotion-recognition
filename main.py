"""
Speech Emotion Recognition using PASe+ (Problem-Agnostic Speech Encoder Plus)

This pipeline extracts features from speech using PASe+ and trains 
an emotion classifier on top of the learned representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_speaker_map(csv_dir):
    """Load speaker info from CSV metadata files.
    Returns dict: wav_id -> speaker_id (age_gender)
    """
    dfs = []
    csv_dir = Path(csv_dir)
    for csv_file in csv_dir.glob('*.csv'):
        df = pd.read_csv(csv_file, encoding='cp949')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    cols = list(df.columns)
    age_col, gender_col = cols[-2], cols[-1]
    df['speaker'] = df[age_col].astype(str) + '_' + df[gender_col].astype(str)
    return dict(zip(df['wav_id'].astype(str), df['speaker']))


def speaker_independent_split(features, labels, paths, speaker_map, test_ratio=0.2, val_ratio=0.1):
    """Split data so that no speaker appears in both train and test/val sets."""
    # Map each sample to its speaker
    speakers = []
    for p in paths:
        wav_id = Path(p).stem
        speakers.append(speaker_map.get(wav_id, f'unknown_{wav_id}'))

    # Get unique speakers and their indices
    from collections import defaultdict
    speaker_to_indices = defaultdict(list)
    for i, spk in enumerate(speakers):
        speaker_to_indices[spk].append(i)

    unique_speakers = list(speaker_to_indices.keys())
    np.random.seed(42)
    np.random.shuffle(unique_speakers)

    # Allocate speakers to test, val, train by cumulative sample count
    total = len(paths)
    test_target = int(total * test_ratio)
    val_target = int(total * val_ratio)

    test_indices, val_indices, train_indices = [], [], []
    test_spk, val_spk, train_spk = [], [], []
    test_count, val_count = 0, 0

    for spk in unique_speakers:
        idx = speaker_to_indices[spk]
        if test_count < test_target:
            test_indices.extend(idx)
            test_count += len(idx)
            test_spk.append(spk)
        elif val_count < val_target:
            val_indices.extend(idx)
            val_count += len(idx)
            val_spk.append(spk)
        else:
            train_indices.extend(idx)
            train_spk.append(spk)

    print(f"  Speakers - Train: {len(train_spk)}, Val: {len(val_spk)}, Test: {len(test_spk)}")

    def select(indices):
        idx = torch.tensor(indices)
        return features[idx], labels[idx]

    train_f, train_l = select(train_indices)
    val_f, val_l = select(val_indices)
    test_f, test_l = select(test_indices)
    return train_f, train_l, val_f, val_l, test_f, test_l


def transcribe_all(data_dir, cache_path='transcriptions.json', whisper_model='small'):
    """Transcribe all wav files using Whisper and cache results.

    Args:
        data_dir: Root directory containing emotion subdirectories with wav files.
        cache_path: Path to save/load transcription cache JSON.
        whisper_model: Whisper model size (default 'small').

    Returns:
        dict: wav_id -> transcribed text
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            transcriptions = json.load(f)
        print(f"[INFO] Loaded cached transcriptions: {len(transcriptions)} entries from {cache_path}")
        return transcriptions

    import whisper

    print(f"[INFO] Loading Whisper '{whisper_model}' model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(whisper_model, device=device)
    print(f"[INFO] Whisper loaded on {device}")

    # Collect all wav files
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.rglob('*.wav'))
    print(f"[INFO] Transcribing {len(wav_files)} files...")

    transcriptions = {}
    for wav_path in tqdm(wav_files, desc="Whisper STT"):
        wav_id = wav_path.stem
        try:
            # Load audio via soundfile (no ffmpeg needed for .wav)
            audio_data, sr = sf.read(str(wav_path), dtype='float32')
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if sr != 16000:
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze(0).numpy()
            # Pad/trim to 30s as Whisper expects
            import whisper.audio as whisper_audio
            audio_data = whisper_audio.pad_or_trim(audio_data)
            result = model.transcribe(audio_data, language='ko')
            transcriptions[wav_id] = result['text'].strip()
        except Exception as e:
            print(f"[WARN] Failed to transcribe {wav_id}: {e}")
            transcriptions[wav_id] = ""

    # Save cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved transcriptions to {cache_path} ({len(transcriptions)} entries)")

    # Unload Whisper to free GPU memory
    del model
    torch.cuda.empty_cache()
    print("[INFO] Whisper model unloaded, GPU memory freed")

    return transcriptions


class PASePlusWrapper:
    """
    Wrapper for PASe+ model
    PASe+ paper: https://arxiv.org/abs/2107.04051
    
    This is a simplified version. In practice, you would:
    1. Clone the official repo: https://github.com/glam-imperial/PASe
    2. Load pretrained weights
    """
    def __init__(self, model_path=None):
        # In real usage, load pretrained PASe+ model
        # For demonstration, using a simple feature extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PASe+ typically outputs 100-dim features per frame
        # This is a placeholder - replace with actual PASe+ model
        self.feature_dim = 100
        
        print(f"[INFO] PASe+ wrapper initialized on {self.device}")
        print("[NOTE] Replace this with actual PASe+ model loading")
        
    def extract_features(self, audio_tensor, sr=16000):
        """
        Extract PASe+ features from audio
        
        Args:
            audio_tensor: torch.Tensor of shape (batch, samples) or (samples,)
            sr: sampling rate
            
        Returns:
            features: torch.Tensor of shape (batch, feature_dim)
        """
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # In real implementation, this would call PASe+ encoder
        # For now, using MFCCs as a placeholder
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=40,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
        )
        
        features_list = []
        for audio in audio_tensor:
            mfcc = mfcc_transform(audio.cpu())  # (n_mfcc, time)
            # Global average pooling
            pooled = mfcc.mean(dim=1)  # (n_mfcc,)
            features_list.append(pooled)
        
        features = torch.stack(features_list)  # (batch, n_mfcc)
        
        # Simulate PASe+ output dimension
        if features.shape[1] != self.feature_dim:
            linear = nn.Linear(features.shape[1], self.feature_dim)
            features = linear(features)
        
        return features


class Wav2Vec2Wrapper:
    """
    Wrapper for facebook/wav2vec2-base pretrained model.
    Outputs 768-dim features via time-axis mean pooling.
    Weights are frozen (inference only).
    """
    def __init__(self):
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 768

        print("[INFO] Loading facebook/wav2vec2-base...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[INFO] Wav2Vec2 loaded on {self.device} (feature_dim={self.feature_dim})")

    def extract_features(self, audio_tensor, sr=16000):
        """
        Args:
            audio_tensor: (batch, samples) or (samples,)
        Returns:
            features: (batch, 768)
        """
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Processor expects list of numpy arrays
        input_values = self.processor(
            audio_tensor.cpu().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            hidden_states = outputs.last_hidden_state  # (batch, time, 768)

        # Mean pooling over time axis
        features = hidden_states.mean(dim=1)  # (batch, 768)
        return features


class EmotionDataset(Dataset):
    """
    Dataset for emotion recognition
    Expects directory structure:
        data/
            angry/
                audio1.wav
                audio2.wav
            happy/
                audio1.wav
                audio2.wav
            ...
    """
    def __init__(self, data_dir, transform=None, target_sr=16000, max_len_sec=5):
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.transform = transform
        self.max_len = target_sr * max_len_sec  # 고정 길이 (샘플 수)
        
        # Get all audio files and labels
        self.audio_files = []
        self.labels = []
        self.emotion_to_idx = {}
        
        emotion_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for idx, emotion_folder in enumerate(emotion_folders):
            emotion = emotion_folder.name
            self.emotion_to_idx[emotion] = idx
            
            audio_files = list(emotion_folder.glob('*.wav'))
            self.audio_files.extend(audio_files)
            self.labels.extend([idx] * len(audio_files))
        
        self.idx_to_emotion = {v: k for k, v in self.emotion_to_idx.items()}
        self.num_classes = len(self.emotion_to_idx)
        
        print(f"[INFO] Loaded {len(self.audio_files)} audio files")
        print(f"[INFO] Emotions: {self.emotion_to_idx}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        data, sr = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Remove channel dimension
        waveform = waveform.squeeze(0)

        # Pad or truncate to fixed length
        if waveform.shape[0] > self.max_len:
            waveform = waveform[:self.max_len]
        elif waveform.shape[0] < self.max_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.shape[0]))

        # Apply transforms
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label, str(audio_path)


class EmotionClassifier(nn.Module):
    """
    Emotion classifier built on top of PASe+ features
    """
    def __init__(self, input_dim=100, hidden_dim=128, num_classes=4, dropout=0.5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class DeepEmotionClassifier(nn.Module):
    """
    Deeper classifier for high-dimensional features (e.g. Wav2Vec2 768-dim)
    768 -> 512 -> 256 -> 128 -> 64 -> num_classes
    """
    def __init__(self, input_dim=768, num_classes=4, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class Wav2Vec2FineTuneModel(nn.Module):
    """End-to-end Wav2Vec2 with last N transformer layers unfrozen for fine-tuning."""

    def __init__(self, num_classes=7, unfreeze_last_n=4, dropout=0.3,
                 model_name="facebook/wav2vec2-base"):
        super().__init__()
        from transformers import Wav2Vec2Model

        print(f"[INFO] Loading {model_name} for fine-tuning...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_dim = self.wav2vec2.config.hidden_size  # 768 for base, 1024 for large

        # Freeze everything first
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers
        total_layers = len(self.wav2vec2.encoder.layers)
        for layer in self.wav2vec2.encoder.layers[total_layers - unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.parameters())
        print(f"[INFO] Trainable: {trainable:,} / {total_p:,} params ({100*trainable/total_p:.1f}%)")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_dim)
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        return self.classifier(pooled)


def finetune_wav2vec2(dataset, split_mode='speaker', data_dir='data/emotions',
                      epochs=15, batch_size=4, grad_accum=4,
                      model_name="facebook/wav2vec2-base", model_label="wav2vec2ft"):
    """End-to-end Wav2Vec2 fine-tuning pipeline."""
    from transformers import AutoFeatureExtractor
    from torch.utils.data import Subset
    from collections import defaultdict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Model
    print(f"\n[Step 1] Initializing {model_label} fine-tune model...")
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2FineTuneModel(num_classes=dataset.num_classes, model_name=model_name).to(device)

    # 2. Split by indices
    print(f"\n[Step 2] Splitting data ({split_mode.upper()})...")
    all_paths = [str(dataset.audio_files[i]) for i in range(len(dataset))]

    if split_mode == 'speaker':
        CSV_DIR = Path(data_dir).parent.parent / '감정 분류를 위한 대화 음성 데이터셋'
        speaker_map = load_speaker_map(CSV_DIR)
        print(f"  Loaded speaker info for {len(speaker_map)} samples")

        speakers = [speaker_map.get(Path(p).stem, f'unknown_{Path(p).stem}') for p in all_paths]
        speaker_to_indices = defaultdict(list)
        for i, spk in enumerate(speakers):
            speaker_to_indices[spk].append(i)
        unique_speakers = list(speaker_to_indices.keys())
        np.random.seed(42)
        np.random.shuffle(unique_speakers)

        total = len(all_paths)
        test_target, val_target = int(total * 0.2), int(total * 0.1)
        train_indices, val_indices, test_indices = [], [], []
        test_count, val_count = 0, 0

        for spk in unique_speakers:
            idx = speaker_to_indices[spk]
            if test_count < test_target:
                test_indices.extend(idx)
                test_count += len(idx)
            elif val_count < val_target:
                val_indices.extend(idx)
                val_count += len(idx)
            else:
                train_indices.extend(idx)
    else:
        indices = list(range(len(dataset)))
        labels_arr = np.array(dataset.labels)
        train_val_idx, test_indices = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels_arr)
        train_indices, val_indices = train_test_split(
            train_val_idx, test_size=0.1/0.8, random_state=42,
            stratify=labels_arr[train_val_idx])

    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # 3. DataLoaders
    def collate_fn(batch):
        waveforms, labels, paths = zip(*batch)
        inputs = processor(
            [w.numpy() for w in waveforms],
            sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values
        return inputs, torch.tensor(labels)

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size,
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 4. Optimizer (discriminative LR: small for wav2vec2, large for classifier)
    w2v_params = [p for p in model.wav2vec2.parameters() if p.requires_grad]
    cls_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {'params': w2v_params, 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': cls_params, 'lr': 1e-3, 'weight_decay': 0.01}
    ])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # 5. Train
    eff_batch = batch_size * grad_accum
    print(f"\n[Step 3] Training (epochs={epochs}, batch={batch_size}x{grad_accum}={eff_batch})")
    print(f"[INFO] Device: {device}")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, (input_values, labels) in enumerate(pbar):
            input_values = input_values.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(input_values)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{total_loss/(step+1):.3f}", acc=f"{100.*correct/total:.1f}%")

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for input_values, labels in val_loader:
                input_values = input_values.to(device)
                labels = labels.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(input_values)
                    loss = criterion(logits, labels)
                v_loss += loss.item()
                _, predicted = logits.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()

        val_loss = v_loss / len(val_loader)
        val_acc = 100. * v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{model_label}.pth')

    print(f"\n[INFO] Best Val Acc: {best_val_acc:.2f}%")

    # 6. Test
    print("\n[Step 4] Evaluating on test set...")
    model.load_state_dict(torch.load(f'best_model_{model_label}.pth', weights_only=True))
    model.eval()

    t_loss, t_correct, t_total = 0, 0, 0
    test_preds, test_true = [], []

    with torch.no_grad():
        for input_values, labels in tqdm(test_loader, desc="Testing"):
            input_values = input_values.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(input_values)
                loss = criterion(logits, labels)
            t_loss += loss.item()
            _, predicted = logits.max(1)
            t_total += labels.size(0)
            t_correct += predicted.eq(labels).sum().item()
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_loss = t_loss / len(test_loader)
    test_acc = 100. * t_correct / t_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 7. Report
    class_names = [dataset.idx_to_emotion[i] for i in range(dataset.num_classes)]
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=class_names))

    # 8. Plots & save
    suffix = f"_{model_label}_{split_mode}"
    plot_training_history(history, save_path=f'training_history{suffix}.png')
    plot_confusion_matrix(test_true, test_preds, class_names=class_names,
                          save_path=f'confusion_matrix{suffix}.png')

    results = {
        'model': model_label, 'split': split_mode,
        'test_accuracy': test_acc, 'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'num_classes': dataset.num_classes,
        'emotion_mapping': dataset.emotion_to_idx,
        'num_train': len(train_indices),
        'num_val': len(val_indices),
        'num_test': len(test_indices)
    }
    results_path = f'results{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Pipeline completed! [{model_label.upper()} / {split_mode.upper()}]")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"\nOutput files:")
    print(f"  - best_model_{model_label}.pth")
    print(f"  - training_history{suffix}.png")
    print(f"  - confusion_matrix{suffix}.png")
    print(f"  - {results_path}")
    print("=" * 60)


def train_multimodal(data_dir='data/emotions', split_mode='speaker',
                     epochs=50, batch_size=32, cache_path='transcriptions.json'):
    """Multimodal emotion recognition: MFCC (40-dim) + KoBERT CLS (768-dim) fusion.

    Pipeline:
        1. Load transcriptions (Whisper cache)
        2. Extract MFCC features (40-dim) from all audio
        3. Extract KoBERT [CLS] features (768-dim) from transcriptions
        4. Concatenate → 808-dim → MLP classifier
    """
    from transformers import AutoTokenizer, AutoModel
    from collections import defaultdict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(data_dir)

    # === Step 1: Load dataset and transcriptions ===
    print("\n[Step 1] Loading dataset and transcriptions...")
    dataset = EmotionDataset(data_dir)
    transcriptions = transcribe_all(data_dir, cache_path=cache_path, whisper_model='medium')

    # === Step 2: Extract MFCC features ===
    print("\n[Step 2] Extracting MFCC features...")
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=40,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
    )

    mfcc_features = []
    labels_list = []
    paths_list = []

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    for waveforms, labels, paths in tqdm(loader, desc="MFCC extraction"):
        # waveforms: (batch, samples)
        mfcc = mfcc_transform(waveforms)  # (batch, n_mfcc, time)
        pooled = mfcc.mean(dim=2)  # (batch, 40)
        mfcc_features.append(pooled)
        labels_list.append(labels)
        paths_list.extend(paths)

    mfcc_features = torch.cat(mfcc_features, dim=0)  # (N, 40)
    all_labels = torch.cat(labels_list, dim=0)
    print(f"  MFCC features: {mfcc_features.shape}")

    # === Step 3: Extract KoBERT features ===
    print("\n[Step 3] Extracting KoBERT [CLS] features...")
    tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    bert_model = AutoModel.from_pretrained("kykim/bert-kor-base").to(device)
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False

    bert_features = []
    bert_batch_size = 64

    # Collect texts in dataset order
    texts = []
    for p in paths_list:
        wav_id = Path(p).stem
        texts.append(transcriptions.get(wav_id, ""))

    for i in tqdm(range(0, len(texts), bert_batch_size), desc="KoBERT extraction"):
        batch_texts = texts[i:i + bert_batch_size]
        # Replace empty strings with a space to avoid tokenizer issues
        batch_texts = [t if t else " " for t in batch_texts]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert_model(**encoded)
            cls = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        bert_features.append(cls.cpu())

    bert_features = torch.cat(bert_features, dim=0)  # (N, 768)
    print(f"  KoBERT features: {bert_features.shape}")

    # Unload BERT to free GPU memory
    del bert_model, tokenizer
    torch.cuda.empty_cache()
    print("[INFO] KoBERT unloaded, GPU memory freed")

    # === Step 4: Concatenate features ===
    print("\n[Step 4] Fusing features...")
    fused_features = torch.cat([mfcc_features, bert_features], dim=1)  # (N, 808)
    print(f"  Fused features: {fused_features.shape}")

    # === Step 5: Split data ===
    print(f"\n[Step 5] Splitting data ({split_mode.upper()})...")
    if split_mode == 'speaker':
        CSV_DIR = data_dir.parent.parent / '감정 분류를 위한 대화 음성 데이터셋'
        speaker_map = load_speaker_map(CSV_DIR)
        print(f"  Loaded speaker info for {len(speaker_map)} samples")
        train_f, train_l, val_f, val_l, test_f, test_l = \
            speaker_independent_split(fused_features, all_labels, paths_list,
                                      speaker_map, test_ratio=0.2, val_ratio=0.1)
    else:
        train_val_f, test_f, train_val_l, test_l = train_test_split(
            fused_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        train_f, val_f, train_l, val_l = train_test_split(
            train_val_f, train_val_l, test_size=0.1/0.8, random_state=42, stratify=train_val_l)

    print(f"  Train: {len(train_f)}, Val: {len(val_f)}, Test: {len(test_f)}")

    # === Step 6: Build MLP classifier ===
    print("\n[Step 6] Building MLP classifier...")
    classifier = nn.Sequential(
        nn.Linear(808, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(128, dataset.num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    # === Step 7: Train ===
    print(f"\n[Step 7] Training (epochs={epochs}, batch_size={batch_size})...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0

    train_dataset = torch.utils.data.TensorDataset(train_f, train_l)
    val_dataset = torch.utils.data.TensorDataset(val_f, val_l)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        # Train
        classifier.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            _, pred = logits.max(1)
            t_total += labs.size(0)
            t_correct += pred.eq(labs).sum().item()

        train_loss = t_loss / len(train_loader)
        train_acc = 100. * t_correct / t_total

        # Validate
        classifier.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for feats, labs in val_loader:
                feats, labs = feats.to(device), labs.to(device)
                logits = classifier(feats)
                loss = criterion(logits, labs)
                v_loss += loss.item()
                _, pred = logits.max(1)
                v_total += labs.size(0)
                v_correct += pred.eq(labs).sum().item()

        val_loss = v_loss / len(val_loader)
        val_acc = 100. * v_correct / v_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
                  f"Val: {val_loss:.4f} / {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), 'best_model_multimodal.pth')

    print(f"\n[INFO] Best Val Acc: {best_val_acc:.2f}%")

    # === Step 8: Test ===
    print("\n[Step 8] Evaluating on test set...")
    classifier.load_state_dict(torch.load('best_model_multimodal.pth', weights_only=True))
    classifier.eval()

    test_dataset = torch.utils.data.TensorDataset(test_f, test_l)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds, test_true = [], []
    te_loss, te_correct, te_total = 0, 0, 0
    with torch.no_grad():
        for feats, labs in test_loader:
            feats, labs = feats.to(device), labs.to(device)
            logits = classifier(feats)
            loss = criterion(logits, labs)
            te_loss += loss.item()
            _, pred = logits.max(1)
            te_total += labs.size(0)
            te_correct += pred.eq(labs).sum().item()
            test_preds.extend(pred.cpu().numpy())
            test_true.extend(labs.cpu().numpy())

    test_loss = te_loss / len(test_loader)
    test_acc = 100. * te_correct / te_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # === Step 9: Report & visualizations ===
    class_names = [dataset.idx_to_emotion[i] for i in range(dataset.num_classes)]
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, target_names=class_names))

    suffix = f"_multimodal_{split_mode}"
    plot_training_history(history, save_path=f'training_history{suffix}.png')
    plot_confusion_matrix(test_true, test_preds, class_names=class_names,
                          save_path=f'confusion_matrix{suffix}.png')

    results = {
        'model': 'multimodal (MFCC+KoBERT)',
        'split': split_mode,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'feature_dim': 808,
        'mfcc_dim': 40,
        'bert_dim': 768,
        'num_classes': dataset.num_classes,
        'emotion_mapping': dataset.emotion_to_idx,
        'num_train': len(train_f),
        'num_val': len(val_f),
        'num_test': len(test_f)
    }
    results_path = f'results{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Multimodal Pipeline completed! [MFCC+KoBERT / {split_mode.upper()}]")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"\nOutput files:")
    print(f"  - best_model_multimodal.pth")
    print(f"  - training_history{suffix}.png")
    print(f"  - confusion_matrix{suffix}.png")
    print(f"  - {results_path}")
    print("=" * 60)


class EmotionRecognitionPipeline:
    """
    End-to-end pipeline for emotion recognition using PASe+
    """
    def __init__(self, pase_model, num_classes=4, device=None, deep=False):
        self.pase = pase_model
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize classifier
        if deep:
            self.classifier = DeepEmotionClassifier(
                input_dim=self.pase.feature_dim,
                num_classes=num_classes
            ).to(self.device)
        else:
            self.classifier = EmotionClassifier(
                input_dim=self.pase.feature_dim,
                num_classes=num_classes
            ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def extract_pase_features(self, dataloader):
        """Extract PASe+ features from entire dataset"""
        features_list = []
        labels_list = []
        paths_list = []
        
        print("[INFO] Extracting PASe+ features...")
        
        with torch.no_grad():
            for waveforms, labels, paths in tqdm(dataloader):
                waveforms = waveforms.to(self.device)
                
                # Extract PASe+ features
                features = self.pase.extract_features(waveforms)
                
                features_list.append(features.cpu())
                labels_list.append(labels)
                paths_list.extend(paths)
        
        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
        
        return all_features, all_labels, paths_list
    
    def train_epoch(self, features, labels):
        """Train for one epoch"""
        self.classifier.train()
        
        # Create dataloader from features
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.classifier(batch_features)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, features, labels):
        """Evaluate the model"""
        self.classifier.eval()
        
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.classifier(batch_features)
                loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_features, train_labels, val_features, val_labels, epochs=50):
        """Train the classifier"""
        print(f"\n[INFO] Training on {self.device}")
        print(f"[INFO] Train samples: {len(train_features)}, Val samples: {len(val_features)}")
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_features, train_labels)
            
            # Validate
            val_loss, val_acc, _, _ = self.evaluate(val_features, val_labels)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
        
        print(f"\n[INFO] Training completed. Best Val Acc: {best_val_acc:.2f}%")
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def plot_training_history(history, save_path='training_history.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """Main execution pipeline"""

    parser = argparse.ArgumentParser(description='Speech Emotion Recognition')
    parser.add_argument('--model', type=str, default='mfcc',
                        choices=['mfcc', 'wav2vec2', 'wav2vec2ft', 'emotion', 'multimodal'],
                        help='Feature extractor: mfcc, wav2vec2 (frozen), wav2vec2ft (fine-tune), emotion (XLSR-emotion fine-tune), multimodal (MFCC+KoBERT fusion)')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'speaker'],
                        help='Data split strategy: random or speaker-independent')
    args = parser.parse_args()

    # Configuration
    DATA_DIR = 'data/emotions'  # Change this to your data directory
    BATCH_SIZE = 16
    EPOCHS = 50
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    model_name = args.model.upper()
    split_name = args.split.upper()
    print("="*60)
    print(f"Speech Emotion Recognition - Feature: {model_name}, Split: {split_name}")
    print("="*60)
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\n[WARNING] Data directory not found: {DATA_DIR}")
        print("[INFO] Creating dummy dataset structure...")
        print("\nExpected structure:")
        print("  data/emotions/")
        print("      angry/")
        print("          audio1.wav")
        print("      happy/")
        print("          audio1.wav")
        print("      sad/")
        print("      neutral/")
        print("\nPlease prepare your dataset and run again.")
        return

    # Multimodal path: MFCC + KoBERT fusion via Whisper STT
    if args.model == 'multimodal':
        train_multimodal(data_dir=DATA_DIR, split_mode=args.split, epochs=EPOCHS)
        return

    # Fine-tuning path: end-to-end training (separate from frozen feature extraction)
    if args.model in ('wav2vec2ft', 'emotion'):
        dataset = EmotionDataset(DATA_DIR)
        if args.model == 'emotion':
            finetune_wav2vec2(
                dataset, split_mode=args.split, data_dir=DATA_DIR,
                model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                model_label="emotion", batch_size=2, grad_accum=8)
        else:
            finetune_wav2vec2(dataset, split_mode=args.split, data_dir=DATA_DIR)
        return

    # 1. Initialize feature extractor
    print(f"\n[Step 1] Initializing {model_name} feature extractor...")
    if args.model == 'wav2vec2':
        pase_model = Wav2Vec2Wrapper()
    else:
        pase_model = PASePlusWrapper()
    
    # 2. Load dataset
    print("\n[Step 2] Loading dataset...")
    dataset = EmotionDataset(DATA_DIR)
    
    # 3. Create data loaders
    print("\n[Step 3] Creating data loaders...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. Initialize pipeline
    print("\n[Step 4] Initializing training pipeline...")
    use_deep = (args.model == 'wav2vec2')
    pipeline = EmotionRecognitionPipeline(
        pase_model=pase_model,
        num_classes=dataset.num_classes,
        deep=use_deep
    )
    
    # 5. Extract PASe+ features
    print("\n[Step 5] Extracting PASe+ features from all audio files...")
    all_features, all_labels, all_paths = pipeline.extract_pase_features(dataloader)
    
    # 6. Split data
    print(f"\n[Step 6] Splitting data ({split_name})...")
    if args.split == 'speaker':
        CSV_DIR = Path(DATA_DIR).parent.parent / '감정 분류를 위한 대화 음성 데이터셋'
        speaker_map = load_speaker_map(CSV_DIR)
        print(f"  Loaded speaker info for {len(speaker_map)} samples")
        train_features, train_labels, val_features, val_labels, test_features, test_labels = \
            speaker_independent_split(all_features, all_labels, all_paths,
                                      speaker_map, test_ratio=TEST_SIZE, val_ratio=VAL_SIZE)
    else:
        # First split: train+val vs test
        train_val_features, test_features, train_val_labels, test_labels = train_test_split(
            all_features, all_labels, test_size=TEST_SIZE, random_state=42, stratify=all_labels
        )
        # Second split: train vs val
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_val_features, train_val_labels,
            test_size=VAL_SIZE/(1-TEST_SIZE),
            random_state=42,
            stratify=train_val_labels
        )

    print(f"  Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # 7. Train classifier
    print("\n[Step 7] Training emotion classifier...")
    pipeline.train(
        train_features, train_labels,
        val_features, val_labels,
        epochs=EPOCHS
    )
    
    # 8. Evaluate on test set
    print("\n[Step 8] Evaluating on test set...")
    test_loss, test_acc, test_preds, test_true = pipeline.evaluate(test_features, test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # 9. Generate classification report
    print("\n[Step 9] Generating classification report...")
    print("\nClassification Report:")
    print(classification_report(
        test_true, test_preds,
        target_names=[dataset.idx_to_emotion[i] for i in range(dataset.num_classes)]
    ))
    
    # 10. Save visualizations
    print("\n[Step 10] Saving visualizations...")
    suffix = f"_{args.model}_{args.split}"
    plot_training_history(pipeline.history, save_path=f'training_history{suffix}.png')
    plot_confusion_matrix(
        test_true, test_preds,
        class_names=[dataset.idx_to_emotion[i] for i in range(dataset.num_classes)],
        save_path=f'confusion_matrix{suffix}.png'
    )

    # 11. Save results
    print("\n[Step 11] Saving results...")
    results = {
        'model': args.model,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'num_classes': dataset.num_classes,
        'emotion_mapping': dataset.emotion_to_idx,
        'num_train': len(train_features),
        'num_val': len(val_features),
        'num_test': len(test_features)
    }

    results_path = f'results{suffix}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("\n" + "="*60)
    print(f"Pipeline completed successfully! [{model_name} / {split_name}]")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"\nOutput files:")
    print(f"  - best_model.pth (model checkpoint)")
    print(f"  - training_history{suffix}.png (training curves)")
    print(f"  - confusion_matrix{suffix}.png (confusion matrix)")
    print(f"  - {results_path} (final results)")
    print("="*60)


if __name__ == '__main__':
    main()
