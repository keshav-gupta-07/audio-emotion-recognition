from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

model = load_model('emotion_recognition_model.h5')
df = pd.read_csv('emotion_features.csv')

X = df.drop(['emotion'], axis=1)
y = df['emotion']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

target_names = [str(i) for i in label_encoder.classes_]
print(classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=0))
