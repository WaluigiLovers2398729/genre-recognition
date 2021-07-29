# USER QUERY FUNCTIONS
# Note: If we have time, we'll most likely want to add a recording option for query instead of solely files.
# So, raw recording --> .wav, otherwise convert file type --> save file --> process to spectrogram .png

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from pydub import AudioSegment
import matplotlib.pyplot as plt
from .model import *
import numpy as np
import librosa
import os

def process_query(model, filename):
    """
    docstring
    """
    # extract relevant data from original audio file
    sound = AudioSegment.from_wav(os.path.join("recognition/data/query/", f"{filename}.wav"))
    sound = sound[1000*40:1000*50]
    temp_filename = str(filename) + "_extracted"
    sound.export(f"recognition/data/query/{temp_filename}.wav", format='wav')
    # load in 5 seconds of audio for this sound file
    y, sr = librosa.load(f"recognition/data/query/{temp_filename}.wav", duration=5)
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
    plt.savefig(f"recognition/data/query/{temp_filename}.png")

    # converts spectrogram image into an array
    loaded = load_img(f"recognition/data/query/{temp_filename}.png", target_size=(288, 432))
    image_arr = img_to_array(loaded)
    print(image_arr.shape)
    # reshapes to the same dimensions & color channels of those from training   
    image_arr = np.reshape(image_arr,(1, 288, 432, 3)) 
    # uses GenreModel to make prediction, /255 scales rgb coefficients down for the model
    predictions = model.predict(image_arr/255)
    # reshape predictions into 9 genre frequencies
    predictions = predictions.reshape((9,)) 
    # gets respective labels for predictions
    best_prediction = np.argmax(predictions)
    # return labels and predictions for plotting
    return best_prediction, predictions

    