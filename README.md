# Genre-Recognition
# Note: We might want to relabel this as genre classification/detection for a more accurate description (also give it a catchy name)

Simple Python Project Template

The basics of creating an installable Python package.

To install this package, in the same directory as setup.py run the command:

pip install -e .

This will install example_project in your Python environment. You can now use it as:

from example_project import returns_one
from example_project.functions_a import hello_world
from example_project.functions_b import multiply_and_sum

To change then name of the project, do the following:

    change the name of the directory example_project/ to your project's name (it must be a valid python variable name, e.g. no spaces allowed)
    change the PROJECT_NAME in setup.py to the same name
    install this new package (pip install -e .)

If you changed the name to, say, my_proj, then the usage will be:

from my_proj import returns_one
from my_proj.functions_a import hello_world
from my_proj.functions_b import multiply_and_sum

You can read more about the basics of creating a Python package here.

# Planning Sheet

Another model pre-trained with GTZAN: https://github.com/Hguimaraes/gtzan.keras
GTZAN Dataset: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification
    * Comprised of 10 main music genres
    * 100 thirty-second audio files for each genre
    * 100 respective mel spectrograms for each of those audio files

THINGS TO MAKE PROJECT UNIQUE
    * Combining with wahzam + visualizer so a single song input could output the song name, song genre, and a visual representation → and maybe add a ui and name it something user-friendly like tune-
        * 1) Genre Classification + Simple Audio Input
        * 2) Swap simple audio input for wahzam song recognition
        * 3) Add live visualizer for the inputted song
        * 4) Add a nice UI 

process_data.py: DATA 
    * Load in dataset

    * Split audio files like in the article to increase accuracy (can use pydub audio segment)
    
    * Audio files → mel spectrograms
        * Might use matplotlib in conjunction with (else just use provided import)
            * import Image 
            * import matplotlib.pyplot as plt 
            * plt.plot(range(10)) plt.savefig('testplot.png') Image.open('testplot.png').save('testplot.jpg','JPEG')

        * (Mel spectrogram will be more effective, if we have time can figure out how to scale our original matplotlib ones to mel)

        * Will have to keep in mind the dimensionality of this, 0 pad to be compatible with convolutions
        
    * We save the data locally, retrieve using ImageDataGenerator for usage with batches 

model.py: MODEL
    * CNN: 5 convolutional layers 
        * 8 filters first layer, 16, 32, 64, 128  - all 3x3 filter size, stride 1
        * 1 dropout layer (to avoid overfitting) 
        * dense layer (with softmax activation)

training.py: TRAINING
    * Will most likely make use of (f1) functions from reading since we’re working with directory files unlike in cogworks
    
main_functions.py: RECOGNITION
    * input song

    * recognition
