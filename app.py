import streamlit as st
import numpy as np
import torch
import torchaudio
import librosa
from sklearn.preprocessing import StandardScaler
import pickle
from torch import nn
import os
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
        st.error(f"Error processing audio: {e}")
        return None

# Predict emotion function
def predict_emotion(audio_file, model_path, scaler_path, le_path, device='cpu'):
    try:
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
        
        # Load and process audio
        audio, sr = librosa.load(audio_file, sr=22050)
        features = extract_features(audio, sr)
        if features is None:
            return None, None
        
        # Scale features
        features_flat = features.reshape(1, -1)
        features_scaled = scaler.transform(features_flat)
        features = features_scaled.reshape(1, features.shape[0], features.shape[1])
        
        # Predict
        features = torch.FloatTensor(features).to(device)
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            pred_emotion = label_encoder.inverse_transform([pred_idx])[0]
        
        return pred_emotion, probabilities
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return None, None

# Streamlit app
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload a WAV audio file to classify the emotion (neutral, calm, happy, sad, angry, fearful, disgust, surprise).")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV audio file", type=["wav"])
    
    # Model and auxiliary files paths
    model_path = "models/emotion_classification_model.pth"
    scaler_path = "models/scaler.pkl"
    le_path = "models/label_encoder.pkl"
    
    # Check if model files exist
    if not all(os.path.exists(path) for path in [model_path, scaler_path, le_path]):
        st.error("Model files not found. Ensure `emotion_classification_model.pth`, `scaler.pkl`, and `label_encoder.pkl` are in the `models/` directory.")
        return
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Play audio
        st.audio(uploaded_file, format="audio/wav")
        
        # Predict emotion
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pred_emotion, probabilities = predict_emotion("temp_audio.wav", model_path, scaler_path, le_path, device)
        
        if pred_emotion is not None:
            st.success(f"Predicted Emotion: **{pred_emotion.capitalize()}**")
            
            # Display confidence scores
            st.write("Confidence Scores:")
            emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            for emotion, prob in zip(emotions, probabilities):
                st.write(f"{emotion.capitalize()}: {prob:.4f}")
            
            # Plot confidence scores
            st.bar_chart(dict(zip(emotions, probabilities)))
        
        # Clean up temporary file
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

if __name__ == "__main__":
    main()