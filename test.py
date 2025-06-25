import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
from torch import nn
import warnings
warnings.filterwarnings('ignore')

# Define CNN-LSTM-Attention model
class CNN_LSTM_Attention_Model(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN_LSTM_Attention_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=187, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.attention = nn.Linear(256, 256)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        attn_output = torch.sum(x * attn_weights, dim=1)
        x = self.dropout(attn_output)
        x = self.fc(x)
        return x

# Feature extraction function
def extract_features(audio, sr=22050, max_length=216):
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        def pad_or_truncate(feature, max_len):
            if feature.shape[1] > max_len:
                return feature[:, :max_len]
            elif feature.shape[1] > 0:
                return np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
            else:
                return np.zeros((feature.shape[0], max_len))
        
        mfcc = pad_or_truncate(mfcc, max_length)
        delta_mfcc = pad_or_truncate(delta_mfcc, max_length)
        delta_delta_mfcc = pad_or_truncate(delta_delta_mfcc, max_length)
        chroma = pad_or_truncate(chroma, max_length)
        mel = pad_or_truncate(mel, max_length)
        zcr = pad_or_truncate(zcr, max_length)
        spectral_contrast = pad_or_truncate(spectral_contrast, max_length)
        
        features = np.concatenate((mfcc, delta_mfcc, delta_delta_mfcc, chroma, mel, zcr, spectral_contrast), axis=0)
        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to get emotion from filename
def get_emotion_from_filename(file):
    try:
        emotion_code = int(file.split('-')[2])
        emotion_map = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
        }
        return emotion_map.get(emotion_code, None)
    except:
        return None

# Function to load and process test data
def load_test_data(test_dir, scaler, label_encoder, max_length=216):
    file_paths = []
    true_labels = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                emotion = get_emotion_from_filename(file)
                if emotion:
                    file_paths.append(file_path)
                    true_labels.append(emotion)
    
    if not file_paths:
        raise ValueError("No valid audio files found in test directory.")
    
    X_test = []
    y_test = []
    for path, label in zip(file_paths, true_labels):
        try:
            audio, sr = librosa.load(path, sr=22050)
            features = extract_features(audio, sr, max_length)
            if features is not None:
                X_test.append(features)
                y_test.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Scale features
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    
    # Encode labels
    y_test_encoded = label_encoder.transform(y_test)
    
    return X_test, y_test_encoded, y_test

# Main test function
def test_model(test_dir, model_path, scaler_path, le_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load scaler and label encoder
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load model
    model = CNN_LSTM_Attention_Model(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and process test data
    X_test, y_test_encoded, y_test = load_test_data(test_dir, scaler, label_encoder)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test_encoded = torch.LongTensor(y_test_encoded).to(device)
    
    # Predict
    predictions = []
    with torch.no_grad():
        for i in range(0, X_test.shape[0], 8):  # Batch size 8
            batch = X_test[i:i+8]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    predictions = np.array(predictions)
    y_test_encoded = y_test_encoded.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(y_test_encoded, predictions)
    weighted_f1 = f1_score(y_test_encoded, predictions, average='weighted')
    per_class_f1 = f1_score(y_test_encoded, predictions, average=None)
    per_class_accuracy = []
    cm = confusion_matrix(y_test_encoded, predictions)
    for i in range(len(label_encoder.classes_)):
        correct = cm[i, i]
        total = cm[i].sum()
        per_class_accuracy.append(correct / total if total > 0 else 0.0)
    
    # Save metrics
    with open('test_metrics.txt', 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        f.write("\nPer-class F1 Scores:\n")
        for i, emotion in enumerate(label_encoder.classes_):
            f.write(f"{emotion}: {per_class_f1[i]:.4f}\n")
        f.write("\nPer-class Accuracies:\n")
        for i, emotion in enumerate(label_encoder.classes_):
            f.write(f"{emotion}: {per_class_accuracy[i]:.4f}\n")
    
    # Save confusion matrix
    with open('test_confusion_matrix.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
    
    # Print metrics
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("\nPer-class F1 Scores:")
    for i, emotion in enumerate(label_encoder.classes_):
        print(f"{emotion}: {per_class_f1[i]:.4f}")
    print("\nPer-class Accuracies:")
    for i, emotion in enumerate(label_encoder.classes_):
        print(f"{emotion}: {per_class_accuracy[i]:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return predictions, y_test

def main():
    parser = argparse.ArgumentParser(description="Test Speech Emotion Recognition model")
    parser.add_argument('--test-dir', type=str, required=True, help='Directory containing test WAV files')
    parser.add_argument('--model_path', type=str, default='models/emotion_classification_model.pth', help='Path to trained model')
    parser.add_argument('--scaler_path', type=str, default='models/scaler.pkl', help='Path to scaler')
    parser.add_argument('--le_path', type=str, default='models/label_encoder.pkl', help='Path to label encoder')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    test_model(args.test_dir, args.model_path, args.scaler_path, args.le_path, device)

if __name__ == "__main__":
    main()