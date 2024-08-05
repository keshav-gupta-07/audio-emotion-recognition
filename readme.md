# Emotion Recognition from Speech

This project aims to develop a machine learning model that can recognize emotions from audio recordings of speech. 
The model uses the RAVDESS dataset, which contains recordings of actors speaking in different emotional tones.

## Project Structure

emotion_recognition/

├── dataset/ # Downloaded RAVDESS dataset
│ ├── Audio_Speech_Actors_01-24/
│ └── Audio_Song_Actors_01-24/
├── venv/ # Virtual environment
├── .gitignore # gitignore file
├── readme.md # Readme
└── requirements.txt # Library requirement file


## Dataset

The dataset used in this project is the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**. It contains 7356 files, including 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent.

**Dataset Source**: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)

### Download the Dataset

1. Visit the [Zenodo RAVDESS page](https://zenodo.org/record/1188976).
2. Download the `Audio_Speech_Actors_01-24.zip` and the `Audio_Song_Actors_01-24.zip` file.
3. Extract the contents into the `dataset` directory within your project.

## Installation

### Step 1: Clone the Repository

```sh
git clone https://github.com/keshav-gupta-07/audio-emotion-recognition.git
cd audio-emotion-recognition
```

### Step 2: Set Up a Virtual Environment

```sh
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```sh
pip install -r requirements.txt
```

### Step 4: Download and Extract the Dataset

1. Visit the Zenodo RAVDESS page.
2. Download the Audio_Speech_Actors_01-24.zip file.
3. Extract the contents into the dataset directory.
