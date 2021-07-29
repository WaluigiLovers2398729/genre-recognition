# DOWNLOADING THE GTZAN DATASET:
# 1) Go to https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
# 2) Click [Download (1 GB)], this only takes a little over a minute
# 3) Unzip and inside the "Data" folder extract "genres_original"
# 4) Manually move "genres_original" to the /recognition/data/raw folder under this genre-recognition project
# Note: A 1 GB dataset is too big to save in the git repository both before and after processing
# the data, so we're storing it locally on our own computers while working on this step

# Put Google Drive folders into the project folder like this:
# genre-recognition
# --recognition
# ----data
# ------audiosamples
# ------raw
# ------spectrograms
# ------testing_data
# ------query
# ------temp (you actually don't need this)

# Note: If processing runs into an error, remember to delete files to try again

# Note: Jazz files in the GTZAN dataset seem to have a running issue, leaving out for now

# Note: Spectrogram step works much faster at the beginning of the function and progressively slows down,
# converting a single genre at a time before restarting the kernel and repeating is the current best way

from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import librosa
import shutil
import random
import os 
genres = ('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock')

def genre_folders():
    """
    [WORKING] creates 9 genre folders in each data directory
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # iterate to make genre folders in each main directory
    for g in genres:
        # make genre folders in audiosamples directory
        path_audio = os.path.join("recognition/data/audiosamples", f"{g}")
        os.makedirs(path_audio)
        # make genre folders in spectrograms directory
        path_audio = os.path.join("recognition/data/spectrograms", f"{g}")
        os.makedirs(path_audio)
        # make genre folders in testing data directory
        path_train = os.path.join("recognition/data/testing_data", f"{g}")
        os.makedirs(path_train)

def split_and_save():
    """
    [WORKING] splits 900 thirty-second clips to 5400 five-second clips and saves to /audiosamples/
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # iterate through genres  
    for g in genres: 
        i = 0
        # print current genre
        print(g)
        # iterate through a list of the 100 sound files in each genre 
        for filename in os.listdir(os.path.join("recognition/data/raw/genres_original", f"{g}")):
            # sound file path
            song = os.path.join(f"recognition/data/raw/genres_original/{g}", f"{filename}")
            i = i + 1
            for w in range(0, 6):
                # calculations for 5 second splits
                # multiply by 1000 since pydub works with milliseconds
                t1 = 5 * (w) * 1000
                t2 = 5 * (w+1) * 1000
                # opens wav sound file
                newAudio = AudioSegment.from_wav(song)
                # slices wav sound file
                new = newAudio[t1:t2]
                # save new sound file in audiosamples folder
                # (i) corresponds to which respective 30-second slice among all 900 original songs
                # (w) corresponds to which 5-second slice among the six from the original 30-seconds
                new.export(f"recognition/data/audiosamples/{g}/{g+str(i)+str(w)}.wav", format="wav")

def to_mel_spectrograms():
    """
    [WORKING] converts the split 5400 five-second clips to mel spectrograms and saves to /spectrograms/
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # iterate through genres
    for g in genres:
        i = 0
        # print current genre
        print(g)
        # iterate through the newly sliced 5400 five-second sound files
        for filename in os.listdir(os.path.join("recognition/data/audiosamples", f"{g}")):
            # sound file path
            song = os.path.join(f"recognition/data/audiosamples/{g}", f"{filename}")
            i = i + 1
            # load in 5 seconds of audio for this sound file
            y, sr = librosa.load(song, duration=5)
            # use time-series and sampling-rate to get mel-spectrogram as np array
            mels = librosa.feature.melspectrogram(y=y, sr=sr)
            # creates a new "Figure" (for displaying data)
            fig = plt.Figure()
            # manually attach a "Canvas" to the "Figure" (for displaying data)
            canvas = FigureCanvas(fig)
            # convert power spectrogram to decibel units using mel-spectrogram and reference 
            # value (scales amplitude to max in mels), then displays data
            p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
            # save to directory as png after figure is displayed
            plt.savefig(f"recognition/data/spectrograms/{g}/{g+str(i)}.png")

def create_sets():
    """
    [WORKING] moves 25% of the 5400 spectrograms to /testing_data/ with the remaining 75% being "training data"
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # files will be moved from this folder to the testing_data folder
    # in other words, spectrograms is our "training data"
    directory = "recognition/data/spectrograms/"
    # iterate through genres
    for g in genres:
        # print current genre
        print(g)
        # list of spectrogram files for this genre
        filenames = os.listdir(os.path.join(directory, f"{g}"))
        # randomly shuffle
        random.shuffle(filenames)
        # 150 random spectrograms from each genre, 1350 in total from nine genres
        # 1350 is 25% of 5400 (standard size for testing data)
        test_files = filenames[0:150]
        # iterate through test_files
        for f in test_files:
            # move them into testing_data
            shutil.move(directory + f"{g}"+ "/" + f, "recognition/data/testing_data/" + f"{g}")

def crop_axes():
    """
    docstring
    """
    for g in genres:

        print("Cropping Training Data: " + str(g))
        for filename in os.listdir(os.path.join(f"recognition/data/spectrograms/", f"{g}")):
            im = Image.open(os.path.join(f"recognition/data/spectrograms/{g}", f"{filename}"))
            cropped = im.crop((55, 45, 388, 241))
            cropped.save(os.path.join(f"recognition/data/spectrograms/{g}", f"{filename}"))

        print("Cropping Validation Data: " + str(g))
        for filename in os.listdir(os.path.join(f"recognition/data/testing_data/", f"{g}")):
            im = Image.open(os.path.join(f"recognition/data/testing_data/{g}", f"{filename}"))
            cropped = im.crop((55, 45, 388, 241))
            cropped.save(os.path.join(f"recognition/data/testing_data/{g}", f"{filename}"))