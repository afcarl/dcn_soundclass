
Developed with Python 2.7 and Tensorflow 1.0
--------------------------------------------

To run this model, 

1) first get the original raw data.
It is messy, uneven lengths, ogg format, some mono, some stero. 

In some origigData directory, 
pro mpt> git clone  https://github.com/karoldvl/ESC-50.git

2) Next we'll massage the data to make it uniform, and convert the audio files to 
spectrogram files in tiff format. 

Navigate to the utils directory, and open the Jupyter (python) notebook,
ESC50_Convert.ipynb

Set the parameters in the first cell, making sure that the directories you want to use exist. 
Run each cell in sequence to create wave files from the ogg files, and then tiff spectrogram files from wav files. 
    
3) Divide your data into two folders, train and validate 
	a) Go to the data directory for the project.
	   Create two new subdirectories there, train and validate
	   Your home directory for this project should now have a directory stucture:
			./data/esc50spect/[subdirectories of sound classes]
			./data/train
			./data/validate

	b) using either cp --parents (or rsync -R on OSX), copy the data you want from esc50spect to either the train folder or the validate folder. 
		Each sound class is divided into five folds, with file names begining with [1-5]
		So for example, to put 20% of the files in train, and 20% in validate, I do this on my Mac:
			go to ./data/esc50spect/
			prompt> rsync -R */[1-4]*.tif ../train
			prompt> rsync -R */5*.tif ../validate

4) The final step is to turn these data into TFRecord files to be read by the tensorflow training code.
	Go to ./data and execute runcmd.txt 
	

5) Now you are ready to back up to the main project directory and train using the TFRecord files for data.

+++++++++++++++

Note: You can test and explore this model with a tiny subset of the ESC-50 sounds provided with the git repository. 
./data2 contains dog and vird sounds, already converted from sounds -> to spectorgrams -> separated into train/ and test/ folders. All you need to do is 
a) navigate to the data2 directory and run the runcmd to convert the data to TFRecords, and then 
b) go to the main project folder and run 
   prompt> python DCNSoundClass.py