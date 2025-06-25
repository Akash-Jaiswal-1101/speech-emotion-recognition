import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from collections import Counter
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

# Augmentation functions
def add_noise(audio, noise_factor=0.035):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * np.max(audio) * noise
    return augmented

def time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=rate)

def time_shift(audio, shift_max=2, sr=22050):
    shift = np.random.randint(-shift_max * sr, shift_max * sr)
    return np.roll(audio, shift)

def pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

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

# Load and preprocess data
def load_and_preprocess_data(data_dirs, labels_map, augment_count=1):
    file_paths = []
    labels = []
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"Error: Directory not found: {data_dir}")
            continue
        print(f"Collecting files from: {data_dir}")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    emotion = get_emotion_from_filename(file)
                    if emotion in labels_map:
                        file_paths.append(file_path)
                        labels.append(labels_map[emotion])
    
    if not file_paths:
        raise ValueError("No valid audio files found.")
    
    # Train-validation split
    train_paths, val_paths, y_train, y_val = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Process training data with augmentation
    X_train = []
    y_train_aug = []
    augmentations = [add_noise, time_stretch, time_shift, pitch_shift]
    for path, label in zip(train_paths, y_train):
        try:
            audio, sr = librosa.load(path, sr=22050)
            features = extract_features(audio, sr)
            if features is not None:
                X_train.append(features)
                y_train_aug.append(label)
            
            # Augment data
            for _ in range(augment_count):
                aug_audio = audio.copy()
                for aug in np.random.choice(augmentations, size=np.random.randint(1, len(augmentations)), replace=False):
                    if aug == pitch_shift:
                        aug_audio = aug(aug_audio, sr, n_steps=np.random.uniform(-1, 1))
                    elif aug == time_stretch:
                        aug_audio = aug(aug_audio, rate=np.random.uniform(0.8, 1.2))
                    elif aug == time_shift:
                        aug_audio = aug(aug_audio, shift_max=2, sr=sr)
                    else:
                        aug_audio = aug(aug_audio)
                features = extract_features(aug_audio, sr)
                if features is not None:
                    X_train.append(features)
                    y_train_aug.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Process validation data (no augmentation)
    X_val = []
    y_val_aug = []
    for path, label in zip(val_paths, y_val):
        try:
            audio, sr = librosa.load(path, sr=22050)
            features = extract_features(audio, sr)
            if features is not None:
                X_val.append(features)
                y_val_aug.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    X_train = np.array(X_train)
    y_train_aug = np.array(y_train_aug)
    X_val = np.array(X_val)
    y_val_aug = np.array(y_val_aug)
    
    # Scale features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_train = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_val = X_val_scaled.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_aug)
    y_val_encoded = label_encoder.transform(y_val_aug)
    
    return X_train, X_val, y_train_encoded, y_val_encoded, scaler, label_encoder

# Training function
def train_model(X_train, X_val, y_train, y_val, num_epochs=100, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = CNN_LSTM_Attention_Model(num_classes=8)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())
        
        train_loss /= (X_train.shape[0] // batch_size)
        train_acc = accuracy_score(train_true, train_preds)
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_loss /= (X_val.shape[0] // batch_size)
        val_acc = accuracy_score(val_true, val_preds)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/emotion_classification_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Final validation metrics
    model.load_state_dict(torch.load('models/emotion_classification_model.pth'))
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for i in range(0, X_val.shape[0], batch_size):
            batch_X = X_val[i:i+batch_size]
            batch_y = y_val[i:i+batch_size]
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(val_true, val_preds)
    weighted_f1 = f1_score(val_true, val_preds, average='weighted')
    per_class_f1 = f1_score(val_true, val_preds, average=None)
    cm = confusion_matrix(val_true, val_preds)
    per_class_accuracy = []
    for i in range(len(np.unique(val_true))):
        correct = cm[i, i]
        total = cm[i].sum()
        per_class_accuracy.append(correct / total if total > 0 else 0.0)
    
    # Save metrics
    with open('val_metrics.txt', 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        f.write("\nPer-class F1 Scores:\n")
        for i, emotion in enumerate(label_encoder.classes_):
            f.write(f"{emotion}: {per_class_f1[i]:.4f}\n")
        f.write("\nPer-class Accuracies:\n")
        for i, emotion in enumerate(label_encoder.classes_):
            f.write(f"{emotion}: {per_class_accuracy[i]:.4f}\n")
    
    with open('val_confusion_matrix.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
    
    print("\nFinal Validation Metrics:")
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

def main():
    # Define dataset path
    dataset_path = '/kaggle/input/speech-emotion'
    data_dirs = [
        f'{dataset_path}/Audio_Speech_Actors_01-24',
        f'{dataset_path}/Audio_Song_Actors_01-24'
    ]
    
    # Define emotion mapping
    labels_map = {
        'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
        'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
    }
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    X_train, X_val, y_train, y_val, scaler, label_encoder = load_and_preprocess_data(data_dirs, labels_map)
    
    # Save scaler and label encoder
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    train_model(X_train, X_val, y_train, y_val, device=device)

if __name__ == "__main__":
    main()