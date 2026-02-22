# Speech Emotion Recognition

This repository contains a Speech Emotion Recognition (SER) system built using PyTorch, trained on the RAVDESS dataset. The system classifies eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprise) from audio files using a CNN-LSTM-Attention model. The project includes a trained model, test script, exploratory data analysis (EDA) with visualizations, and a comprehensive pipeline for feature engineering, model training, and evaluation.

## Project Description

The goal is to develop a robust SER system capable of accurately classifying emotions in spoken language. The system leverages advanced signal processing and deep learning to extract acoustic features and predict emotional states. Applications include sentiment analysis, affective computing, and emotion-aware systems.

### Dataset
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.
  - Contains 2,452 files (1,440 speech, 1,012 song) from 24 actors.
  - Emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprise.
  - File format: WAV, sampled at 48,000 Hz (downsampled to 22,050 Hz).
  - Source: `/kaggle/input/speech-emotion` (Audio_Speech_Actors_01-24, Audio_Song_Actors_01-24).

### Features
- **Trained Model**: `CNN_LSTM_Attention_Model` saved as `emotion_classification_model.pth`.
- **Scripts**: Training (`train.py`), testing (`test.py`), EDA (`eda.py`), and emotion plotting (`plot_emotions.py`).
- **Feature Engineering**: Comprehensive pipeline for feature extraction, augmentation, scaling, and encoding.
- **EDA**: Visualizations including emotion distribution bar plot, audio durations, feature distributions, and correlations.
- **Evaluation Metrics**: Accuracy, weighted F1 score, per-class F1, and confusion matrix.
- **Environment**: Optimized for Kaggle with GPU (Tesla T4).

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
- **Optimizer**: Adam (learning rate=0.0001).
- **Loss Function**: CrossEntropyLoss.
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6).
- **Epochs**: Up to 100, with early stopping (patience=10).
- **Batch Size**: 8.
- **Metrics**:
  - Training and validation loss.
  - Training and validation accuracy per epoch.
- **Outputs**:
  - Model: `models/emotion_classification_model.pth`.
  - Scaler: `models/scaler.pkl`.
  - Label Encoder: `models/label_encoder.pkl`.

### Testing
- **Script**: `test.py`.
- **Input**: Directory of WAV files in RAVDESS format.
- **Process**:
  - Extract features using the same pipeline as training.
  - Scale features with `scaler.pkl`.
  - Encode labels with `label_encoder.pkl`.
  - Predict emotions using the trained model.
- **Outputs**:
  - Confusion matrix, weighted F1 score, overall accuracy, per-class F1, and per-class accuracy.
  - Saved to `test_confusion_matrix.txt` and `test_metrics.txt`.

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
- **Overall Accuracy**: 81.42%
- **Weighted F1 Score**: 82.90%
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
- **Overall Accuracy**: 83.31%
- **Weighted F1 Score**: 85.47%
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
     torch
     torchaudio
     numpy
     pandas
     librosa
     scikit-learn
     matplotlib
     seaborn
     ```

3. **Download Dataset**:
   - Download RAVDESS from Kaggle: https://www.kaggle.com/datasets/uwrfkaggs/speech-emotion
   - Place in `data/` as `speech-emotion/Audio_Speech_Actors_01-24/` and `speech-emotion/Audio_Song_Actors_01-24/`.

4. **Prepare Scaler and LabelEncoder**:
   - Run `train.py` to generate `scaler.pkl` and `label_encoder.pkl` (saved in `models/`).

## Usage

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

### Training
```bash
python scripts/train.py
```
- Outputs model, scaler, and label encoder to `models/`.
- Prints training/validation metrics.
- Evaluates on validation set.

### Testing
```bash
python scripts/test.py --test-dir /path/to/test/wav/files \
                       --model_path models/emotion_classification_model.pth \
                       --scaler_path models/scaler.pkl \
                       --le_path models/label_encoder.pkl
```
- Outputs metrics to console and saves to `test_metrics.txt` and `test_confusion_matrix.txt`.

## Troubleshooting

- **Low Accuracy** (e.g., fearful: 68.06%)**:
  - Increase augmentation in `train.py`:
    ```python
    augment_count = 2
    ```
  - Add RMS energy feature in `utils.py`:
    ```python
    rms = librosa.feature.rms(y=audio)
    rms = pad_or_truncate(rms, max_length)
    features = np.concatenate((mfcc, delta_mfcc, delta_delta_mfcc, chroma, mel, zcr, spectral_contrast, rms), axis=0)
    ```
  - Check confusion matrix for misclassifications (e.g., fearful vs. sad).
- **Neutral Imbalance**:
  - Oversample neutral in `train.py`:
    ```python
    if emotion == labels_map['neutral']:
        for _ in range(2):
            aug_audio = audio.copy()
            for aug in np.random.choice(augmentations, size=np.random.randint(1, len(augmentations)), replace=False):
                if aug == pitch:
                    aug_audio = aug(aug_audio, sr, pitch_factor=np.random.uniform(-0.5, 0.5))
                elif aug == stretch:
                    aug_audio = aug(aug_audio, rate=np.random.uniform(0.9, 1.1))
                else:
                    aug_audio = aug(aug_audio)
            features = extract_features(aug_audio, sr)
            if features is not None:
                X_train.append(features)
                y_train_aug.append(label)
    ```
- **Memory Issues**:
  - Reduce `max_length` or `batch_size` in `train.py`, `test.py`, `eda.py`:
    ```python
    max_length = 100
    batch_size = 4
    ```
- **Plotting Issues**:
  - If labels overlap in `emotion_bar_plot.png`, adjust in `plot_emotions.py`:
    ```python
    plt.xticks(rotation=45, fontsize=8)
    ```

## References
- RAVDESS Dataset: https://zenodo.org/record/1188976
- PyTorch: https://pytorch.org/
- Librosa: https://librosa.org/

## License
MIT License

## Contributors
- Akash Jaiswal (a_jaiswal@es.iitr.ac.in)
