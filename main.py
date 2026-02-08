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
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


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
            mfcc = mfcc_transform(audio)  # (n_mfcc, time)
            # Global average pooling
            pooled = mfcc.mean(dim=1)  # (n_mfcc,)
            features_list.append(pooled)
        
        features = torch.stack(features_list)  # (batch, n_mfcc)
        
        # Simulate PASe+ output dimension
        if features.shape[1] != self.feature_dim:
            linear = nn.Linear(features.shape[1], self.feature_dim)
            features = linear(features)
        
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
    def __init__(self, data_dir, transform=None, target_sr=16000):
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.transform = transform
        
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
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Remove channel dimension
        waveform = waveform.squeeze(0)
        
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


class EmotionRecognitionPipeline:
    """
    End-to-end pipeline for emotion recognition using PASe+
    """
    def __init__(self, pase_model, num_classes=4, device=None):
        self.pase = pase_model
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize classifier
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
    
    # Configuration
    DATA_DIR = 'data/emotions'  # Change this to your data directory
    BATCH_SIZE = 16
    EPOCHS = 50
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    print("="*60)
    print("Speech Emotion Recognition using PASe+")
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
    
    # 1. Initialize PASe+ model
    print("\n[Step 1] Initializing PASe+ model...")
    pase_model = PASePlusWrapper()
    
    # 2. Load dataset
    print("\n[Step 2] Loading dataset...")
    dataset = EmotionDataset(DATA_DIR)
    
    # 3. Create data loaders
    print("\n[Step 3] Creating data loaders...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 4. Initialize pipeline
    print("\n[Step 4] Initializing training pipeline...")
    pipeline = EmotionRecognitionPipeline(
        pase_model=pase_model,
        num_classes=dataset.num_classes
    )
    
    # 5. Extract PASe+ features
    print("\n[Step 5] Extracting PASe+ features from all audio files...")
    all_features, all_labels, all_paths = pipeline.extract_pase_features(dataloader)
    
    # 6. Split data
    print("\n[Step 6] Splitting data into train/val/test sets...")
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
    plot_training_history(pipeline.history)
    plot_confusion_matrix(
        test_true, test_preds,
        class_names=[dataset.idx_to_emotion[i] for i in range(dataset.num_classes)]
    )
    
    # 11. Save results
    print("\n[Step 11] Saving results...")
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'num_classes': dataset.num_classes,
        'emotion_mapping': dataset.emotion_to_idx,
        'num_train': len(train_features),
        'num_val': len(val_features),
        'num_test': len(test_features)
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nOutput files:")
    print("  - best_model.pth (model checkpoint)")
    print("  - training_history.png (training curves)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("  - results.json (final results)")
    print("="*60)


if __name__ == '__main__':
    main()
