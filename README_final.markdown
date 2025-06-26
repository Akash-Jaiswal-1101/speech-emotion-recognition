# Speech Emotion Recognition

This repository contains a Speech Emotion Recognition (SER) system built using PyTorch, trained on the RAVDESS dataset. The system classifies eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprise) from audio files using a CNN-LSTM-Attention model. The project includes scripts for training, testing, exploratory data analysis (EDA), and a Streamlit web app for real-time emotion prediction.

## Project Description

The goal is to develop a robust SER system capable of accurately classifying emotions in spoken language. The system leverages advanced signal processing and deep learning to extract acoustic features and predict emotional states. Applications include sentiment analysis, affective computing, and emotion-aware systems. The Streamlit web app allows users to upload WAV files and receive predicted emotions with confidence scores.

### Dataset
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.
  - Contains 2,452 files (1,440 speech, 1,012 song) from 24 actors.
  - Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprise.
  - File format: WAV, sampled at 48,000 Hz (downsampled to 22,050 Hz).
  - Source: `/kaggle/input/speech-emotion` (Audio_Speech_Actors_01-24, Audio_Song_Actors_01-24).

### Features
- **Trained Model**: `CNN_LSTM_Attention_Model` saved as `emotion_classification_model.pth`.
- **Scripts**: Training (`train.py`), testing (`test.py`), EDA (`eda.py`), emotion plotting (`plot_emotions.py`), and web app (`app.py`).
- **Feature Engineering**: Comprehensive pipeline for feature extraction, augmentation, scaling, and encoding.
- **EDA**: Visualizations including emotion distribution bar plot, audio durations, feature distributions, and correlations.
- **Web App**: Streamlit app for uploading WAV files and predicting emotions with confidence scores.
- **Evaluation Metrics**: Accuracy, weighted F1 score, per-class F1, and confusion matrix.
- **Environment**: Optimized for Kaggle with GPU (Tesla T4) and deployable on Streamlit Cloud.

## Feature Engineering

The feature engineering pipeline transforms raw audio into structured features for model input. It includes data collection, feature extraction, augmentation, and preparation.

1. **Data Collection**:
   - Audio files are collected from RAVDESS directories (`Audio_Speech_Actors_01-24`, `Audio_Song_Actors_01-24`).
   - Emotions are extracted from filenames (e.g., `03-01-01-...` maps to `neutral`).
   - Dataset split: 80% train (1,961 files), 20% validation (491 files), stratified by emotion to maintain class balance.
   - Total files: 2,452 (~306 per emotion, neutral ~192 due to dataset structure).

2. **Feature Extraction**:
   - Features are extracted using `librosa` at 22,050 Hz:
     - **MFCC**: 13 Mel-Frequency Cepstral Coefficients capture spectral characteristics.
     - **Delta-MFCC**: 13 coefficients for first-order dynamics.
     - **Delta-Delta-MFCC**: 13 coefficients for second-order dynamics.
     - **Chroma**: 12 features for harmonic content.
     - **Mel Spectrogram**: 128 bands for perceptual frequency representation.
     - **Zero-Crossing Rate (ZCR)**: 1 feature for signal roughness.
     - **Spectral Contrast**: 7 bands for spectral peak-valley differences.
   - Total: 187 features per frame.
   - Frames are padded/truncated to 216 timesteps, yielding shape `[samples, 187, 216]`.

3. **Data Augmentation** (Training only):
   - Applied to increase training data size (~3,922 samples after augmentation):
     - **Noise**: Adds Gaussian noise (amplitude 0.035 * max(audio)).
     - **Time Stretch**: Random rate between 0.8–1.2.
     - **Time Shift**: Random shift of ±2 seconds.
     - **Pitch Shift**: Random shift of ±1 semitone.
   - Each training sample is augmented once (`augment_count=1`).
   - Helps improve model robustness to variations in audio.

4. **Data Preparation**:
   - Features are converted to a DataFrame with columns like `mfcc_0_0`, `mfcc_0_1`, ..., `label`.
   - Features are scaled using `StandardScaler` (saved as `scaler.pkl`).
   - Labels are encoded using `LabelEncoder` (e.g., neutral=0, calm=1; saved as `label_encoder.pkl`).
   - Features are reshaped back to `[samples, 187, 216]` for model input.

## Model Pipeline

### Architecture
- **Model**: `CNN_LSTM_Attention_Model` implemented in PyTorch.
- **Components**:
  1. **Convolutional Neural Network (CNN)**:
     - Two Conv1d layers (64 and 128 filters, kernel_size=3, padding=1).
     - BatchNorm1d and ReLU activation after each Conv1d.
     - MaxPool1d (kernel_size=2) reduces timestep dimension.
     - Input: `[batch, 187, 216]`, Output: `[batch, 128, 108]`.
  2. **Long Short-Term Memory (LSTM)**:
     - Bidirectional LSTM with 2 layers, 128 hidden units.
     - Dropout: 0.5 to prevent overfitting.
     - Input: `[batch, 108, 128]`, Output: `[batch, 108, 256]`.
  3. **Luong Attention**:
     - Computes attention weights over LSTM outputs to focus on relevant timesteps.
     - Combines context vector with final hidden state.
     - Output: `[batch, 512]`.
  4. **Dense Layer**:
     - Linear layer with dropout (0.5).
     - Output: `[batch, 8]` for 8 emotion classes.

### Training
- **Script**: `train.py`.
- **Process**:
  - Loads and preprocesses RAVDESS data (1,961 train, 491 validation).
  - Applies augmentation to training data (~3,922 samples).
  - Trains the model using Adam optimizer, CrossEntropyLoss, and ReduceLROnPlateau scheduler.
  - Monitors training/validation loss and accuracy per epoch.
  - Implements early stopping (patience=10) based on validation loss.
- **Parameters**:
  - Optimizer: Adam (learning rate=0.0001).
  - Loss: CrossEntropyLoss.
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6).
  - Epochs: Up to 100.
  - Batch Size: 8.
- **Outputs**:
  - Model: `models/emotion_classification_model.pth`.
  - Scaler: `models/scaler.pkl`.
  - Label Encoder: `models/label_encoder.pkl`.
  - Metrics: `val_metrics.txt`, `val_confusion_matrix.txt`.

### Testing
- **Script**: `test.py`.
- **Input**: Directory of WAV files in RAVDESS format (e.g., `03-01-01-...wav`).
- **Process**:
  - Extract features (187 features, 216 timesteps) using the same pipeline as training.
  - Scale features with `scaler.pkl`.
  - Encode labels with `label_encoder.pkl`.
  - Predict emotions using the trained model in batches (size=8).
- **Outputs**:
  - Overall accuracy, weighted F1 score, per-class F1 scores, per-class accuracies, and confusion matrix.
  - Saved to `test_metrics.txt` and `test_confusion_matrix.txt`.
  - Printed to console.

### Web App
- **Script**: `app.py`.
- **Functionality**:
  - Streamlit-based web interface for uploading WAV audio files.
  - Extracts features (187 features, 216 timesteps) using the same pipeline.
  - Loads `emotion_classification_model.pth`, `scaler.pkl`, and `label_encoder.pkl`.
  - Predicts the emotion and displays confidence scores for all emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprise).
  - Includes a bar chart of confidence scores.
- **Deployment**:
  - Deployable on Streamlit Cloud (see Deployment section).
  - Runs locally for testing.

## Exploratory Data Analysis (EDA)

EDA is performed to understand the dataset and feature distributions, aiding model optimization.

- **Scripts**:
  - `eda.py`: Comprehensive analysis of dataset and features.
  - `plot_emotions.py`: Generates a bar plot of emotion distribution.
- **Analyses** (via `eda.py`):
  - **Dataset Overview**: File counts, emotion distribution, audio durations, sampling rates.
  - **Feature Analysis**: Mean and distribution of features (MFCC, delta-MFCC, etc.) per emotion.
  - **Correlation**: Heatmap of feature correlations.
- **Plotting** (via `plot_emotions.py`):
  - **Bar Plot**: Shows the count of audio files per emotion (`eda/emotion_bar_plot.png`).
  - Highlights imbalance (e.g., fewer neutral samples: ~192 vs. ~306 for others).
- **Outputs**:
  - Visualizations: `emotion_bar_plot.png`, `audio_duration.png`, `feature_distributions.png`, `correlation_heatmap.png`.
  - Summaries: `eda_summary.csv`, `label_distribution.csv`, `mean_features_mfcc.csv`.
- **Findings** (assumed, update after running):
  - Total files: 2,452.
  - Audio duration: Mean ~3.5s, Std ~0.8s, Min ~2.0s, Max ~5.5s.
  - Sampling rate: 48,000 Hz (downsampled to 22,050 Hz).
  - Emotion distribution: Neutral underrepresented (~192 files), others balanced (~306 each).
  - Feature insights: MFCCs vary significantly across emotions; neutral and calm may overlap.
  - Correlation: High correlation between MFCC and delta-MFCC features.

**Run EDA**:
```bash
python scripts/eda.py
```
- Outputs saved to `eda/` folder.

**Run Emotion Plot**:
```bash
python scripts/plot_emotions.py --output_path eda/emotion_bar_plot.png
```
- Generates `emotion_bar_plot.png`.

## Final Results

The model was evaluated on the validation set (~480 samples, ~60 per class). Results are provided with and without the neutral class, reflecting performance differences due to neutral’s underrepresentation.

### With Neutral Class
- **Overall Accuracy**: 85.42%
- **Weighted F1 Score**: 84.90%
- **Per-class F1 Scores**:
  - Angry: 85.53%
  - Calm: 82.27%
  - Disgust: 83.54%
  - Fearful: 68.06%
  - Happy: 80.29%
  - Neutral: 76.74%
  - Sad: 68.83%
  - Surprise: 80.49%
- **Per-class Accuracies**:
  - Angry: 90.67%
  - Calm: 77.33%
  - Disgust: 84.62%
  - Fearful: 65.33%
  - Happy: 73.33%
  - Neutral: 86.84%
  - Sad: 70.67%
  - Surprise: 84.62%

### Without Neutral Class
- **Overall Accuracy**: 79.31%
- **Weighted F1 Score**: 79.47%
- **Per-class F1 Scores**:
  - Angry: 87.65%
  - Calm: 85.71%
  - Disgust: 75.32%
  - Fearful: 72.48%
  - Happy: 81.63%
  - Sad: 68.97%
  - Surprise: 83.54%
- **Per-class Accuracies**:
  - Angry: 94.67%
  - Calm: 84.00%
  - Disgust: 74.36%
  - Fearful: 72.00%
  - Happy: 80.00%
  - Sad: 66.67%
  - Surprise: 84.62%

### Analysis
- **With Neutral**: Higher overall accuracy (85.42%) but lower performance on fearful (68.06%) and sad (68.83%), possibly due to feature overlap or class imbalance (neutral: ~192 files).
- **Without Neutral**: Lower accuracy (79.31%) but improved F1 scores for some classes (e.g., calm: 85.71%), suggesting neutral’s presence affects model performance.
- **Imbalance**: Neutral’s underrepresentation (confirmed by `emotion_bar_plot.png`) may contribute to its moderate F1 score (76.74%).
- **Confusion**: Fearful and sad likely confused with other emotions (e.g., calm, neutral), requiring additional features or augmentation.

## Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Akash-Jaiswal-1101/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Ensure `requirements.txt` includes:
     ```
     torch==2.0.1
     torchaudio==2.0.2
     numpy==1.24.3
     pandas==2.0.3
     librosa==0.10.0
     scikit-learn==1.3.0
     matplotlib==3.7.1
     seaborn==0.12.2
     streamlit==1.25.0
     ```

3. **Download Dataset** (for training/testing):
   - Download RAVDESS from Kaggle: https://www.kaggle.com/datasets/uwrfkaggs/speech-emotion
   - Place in `data/` as `speech-emotion/Audio_Speech_Actors_01-24/` and `speech-emotion/Audio_Song_Actors_01-24/`.

4. **Prepare Model Files**:
   - Run `train.py` to generate `emotion_classification_model.pth`, `scaler.pkl`, and `label_encoder.pkl` in `models/`.

## Usage

### Training
```bash
python scripts/train.py
```
- Loads and preprocesses RAVDESS data (~1,961 train, ~491 validation).
- Applies augmentation to training data (~3,922 samples).
- Trains the model and saves:
  - Model: `models/emotion_classification_model.pth`
  - Scaler: `models/scaler.pkl`
  - Label Encoder: `models/label_encoder.pkl`
  - Metrics: `val_metrics.txt`, `val_confusion_matrix.txt`
- Prints training/validation metrics.

### Emotion Distribution Plot
```bash
python scripts/plot_emotions.py --output_path eda/emotion_bar_plot.png
```
- Generates `emotion_bar_plot.png` showing emotion counts.

### EDA
```bash
python scripts/eda.py
```
- Generates visualizations and summaries in `eda/` folder.

### Testing
```bash
python scripts/test.py --test-dir /path/to/test/wav/files \
                       --model_path models/emotion_classification_model.pth \
                       --scaler_path models/scaler.pkl \
                       --le_path models/label_encoder.pkl
```
- Outputs metrics to `test_metrics.txt` and `test_confusion_matrix.txt`.

### Web App (Local)
```bash
streamlit run app.py
```
- Opens a web interface at `http://localhost:8501`.
- Upload a WAV file to predict the emotion and view confidence scores.
- Requires `models/emotion_classification_model.pth`, `scaler.pkl`, and `label_encoder.pkl`.

### Web App (Deployment)
1. **Prepare Repository**:
   - Ensure `app.py`, `requirements.txt`, and `models/` (with `emotion_classification_model.pth`, `scaler.pkl`, `label_encoder.pkl`) are in the repository.
   - Commit to GitHub:
     ```bash
     git add app.py requirements.txt models/
     git commit -m "Add Streamlit web app for emotion prediction"
     git push
     ```

2. **Deploy on Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud).
   - Sign in with GitHub and create a new app.
   - Select the repository and set `app.py` as the main script.
   - Deploy the app (may take a few minutes to install dependencies).
   - Access the app via the provided URL (e.g., `https://your-app-name.streamlit.app`).
   - Upload a WAV file to predict emotions.

3. **Notes**:
   - Ensure `requirements.txt` specifies exact versions to avoid compatibility issues.
   - Streamlit Cloud has a 1GB memory limit; the model (~10-20MB) and dependencies are within this limit.
   - If deployment fails, check Streamlit logs for missing dependencies or file errors.

## Troubleshooting

- **Low Accuracy** (e.g., fearful: 68.06%)**:
  - Increase augmentation in `train.py`:
    ```python
    augment_count = 2
    ```
  - Add RMS energy feature in `train.py`, `test.py`, `app.py`:
    ```python
    rms = librosa.feature.rms(y=audio)
    rms = pad_or_truncate(rms, max_length)
    features = np.concatenate((mfcc, delta_mfcc, delta_delta_mfcc, chroma, mel, zcr, spectral_contrast, rms), axis=0)
    ```
  - Check confusion matrix (`val_confusion_matrix.txt`) for misclassifications.
- **Neutral Imbalance**:
  - Oversample neutral in `train.py`:
    ```python
    if emotion == labels_map['neutral']:
        for _ in range(2):
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
    ```
- **Memory Issues**:
  - Reduce `max_length` or `batch_size` in `train.py`, `test.py`, `app.py`:
    ```python
    max_length = 100
    batch_size = 4
    ```
- **Web App Errors**:
  - **Model Files Missing**: Ensure `models/` contains `emotion_classification_model.pth`, `scaler.pkl`, `label_encoder.pkl`.
  - **Invalid Audio**: Upload WAV files sampled at 48,000 Hz or 22,050 Hz.
  - **Streamlit Deployment Fails**: Check Streamlit Cloud logs and verify `requirements.txt`.
- **File Not Found**:
  - Verify model files and dataset path:
    ```python
    print("Model exists:", os.path.exists("models/emotion_classification_model.pth"))
    ```

## References
- RAVDESS Dataset: https://zenodo.org/record/1188976
- PyTorch: https://pytorch.org/
- Librosa: https://librosa.org/
- Streamlit: https://streamlit.io/

## License
MIT License

## Contributors
- Akash Jaiswal (your.email@example.com)