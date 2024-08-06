import os
import numpy as np
import pandas as pd
import librosa


def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        print(e)
        return None
    return mfccs_scaled


data_path = "dataset/Audio_Speech_Actors_01-24"

features = []
labels = []

for subdir, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)
            print(f"Processing file:{file_path}")  
            feature = extract_features(file_path)
            if feature is not None:
                try:
                    emotion = int(
                        file.split("-")[2]
                    )
                    features.append(feature)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

df = pd.DataFrame(features)
df["emotion"] = labels

df.to_csv("emotion_features.csv", index=False)
