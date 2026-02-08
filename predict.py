"""
Quick start example for speech emotion recognition

This script demonstrates how to:
1. Load a pretrained model
2. Make predictions on new audio files
3. Visualize predictions
"""

import torch
import torchaudio
from main import PASePlusWrapper, EmotionClassifier
import numpy as np


def predict_emotion(audio_path, model_path='best_model.pth'):
    """
    Predict emotion from a single audio file
    
    Args:
        audio_path: Path to audio file (.wav)
        model_path: Path to trained model checkpoint
        
    Returns:
        predicted_emotion: String of predicted emotion
        confidence: Confidence score (0-1)
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Emotion mapping (must match training)
    idx_to_emotion = {
        0: 'angry',
        1: 'happy',
        2: 'neutral',
        3: 'sad'
    }
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono and resample
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)
    
    # Extract PASe+ features
    pase = PASePlusWrapper()
    with torch.no_grad():
        features = pase.extract_features(waveform.unsqueeze(0), sr=16000)
    
    # Load classifier
    classifier = EmotionClassifier(
        input_dim=100,
        num_classes=len(idx_to_emotion)
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.eval()
    
    # Predict
    features = features.to(device)
    with torch.no_grad():
        outputs = classifier(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_emotion = idx_to_emotion[predicted.item()]
    confidence_score = confidence.item()
    
    # Print all probabilities
    print(f"\n{'='*50}")
    print(f"Audio: {audio_path}")
    print(f"{'='*50}")
    print(f"Predicted Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence_score*100:.2f}%")
    print(f"\nAll Probabilities:")
    for idx, prob in enumerate(probabilities[0]):
        emotion = idx_to_emotion[idx]
        print(f"  {emotion:10s}: {prob.item()*100:5.2f}%")
    print(f"{'='*50}\n")
    
    return predicted_emotion, confidence_score


def batch_predict(audio_dir, model_path='best_model.pth'):
    """
    Predict emotions for all audio files in a directory
    
    Args:
        audio_dir: Directory containing .wav files
        model_path: Path to trained model
        
    Returns:
        results: List of (filename, emotion, confidence) tuples
    """
    from pathlib import Path
    
    audio_files = list(Path(audio_dir).glob('*.wav'))
    results = []
    
    print(f"Processing {len(audio_files)} audio files...\n")
    
    for audio_file in audio_files:
        emotion, confidence = predict_emotion(str(audio_file), model_path)
        results.append((audio_file.name, emotion, confidence))
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"{'Filename':<30} {'Emotion':<10} {'Confidence':<10}")
    print("-"*70)
    for filename, emotion, conf in results:
        print(f"{filename:<30} {emotion:<10} {conf*100:>5.2f}%")
    print("="*70)
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python predict.py path/to/audio.wav")
        print("  Batch mode:   python predict.py path/to/audio_directory/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    from pathlib import Path
    if Path(path).is_file():
        # Single file prediction
        predict_emotion(path)
    elif Path(path).is_dir():
        # Batch prediction
        batch_predict(path)
    else:
        print(f"Error: {path} is not a valid file or directory")
