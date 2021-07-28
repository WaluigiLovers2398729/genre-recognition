import matplotlib.pyplot as plt
# INPUTTING A SONG AND OUTPUTTING GENRE RECOGNITION

def input_query():
    """
    docstring
    """
    input file # assumes in whatever folder

    recording # somehow (???) use mic to take in .mp3 or .wav
    # if not converted, convert to .wav
    # put in the folder

# this is pseudo code basically 
def process_query(filename):
    """
    docstring
    """
    # get filepath given filename
    song = os.path.join("[FILEPATH]/", f"{filename}.wav")
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
    plt.savefig(f"[FILEPATH]/{filename}.png")

def classify_query():   
    """
    docstring
    """
    # converts spectrogram image into an array
    image_arr = img_to_array(image) 
    # reshapes to the same dimensions & color channels of those from training   
    image_arr = np.reshape(image_arr,(1,288,432,4))
    # uses GenreModel to make prediction, /255 scales rgb coefficients down for the model
    predictions = GenreModel.predict(image_arr/255)
    # reshape predictions into 9 genre frequencies
    predictions = predictions.reshape((9,)) 
    # gets respective labels for predictions
    prediction_labels = np.argmax(predictions)
    # return labels and predictions for plotting
    return prediction_labels, predictions

"""
FOR JUPYTER NOTEBOOK
--------------------

# indices for color mapping
color_data = [1,2,3,4,5,6,7,8,9]
# use tab10 color mapping (stand out well against white background)
my_cmap = cm.get_cmap('tab10')

# graph setup, including bar sizes
fig, ax= plt.subplots(figsize=(6,4.5))
ax.bar(x=prediction_labels, height=predictions, color=my_cmap(color_data))

# makes x-axis genre labels tilted 45 degrees
plt.xticks(rotation=45)
# graph title
ax.set_title("Probability Distribution Of The Given Song Over Different Genres")

# plot graph
plt.show()
st.pyplot(fig)
"""