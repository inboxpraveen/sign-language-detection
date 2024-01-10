# Sign Language Detection
This project aims to develop a machine learning model for detecting sign language. Utilizing a Sequential model built with Long Short-Term Memory (LSTM) layers in Keras, the project translates sign language into text or speech, enhancing communication accessibility for individuals who are deaf or hard of hearing.

### How to run this project in your system?

1. Make sure you have Python 3.11.x installed in your system.
2. Ensure Python is in your environment. Type "***python --version***" in command prompt to check.
3. Setup Virtual Environment in your system so nothing breaks. To install virtual environment, run "***pip install vritualenv***". 
4. Once virtual environment is installed, then go to your project folder, and then open command prompt here. 
5. In the opened command prompt, create a new virtual environment using the command - "***python -m venv my-env***"
6. Step 5 will create a folder called "my-env" inside your project folder. Now we have created the virtual environment. We have to activate it. To activate, type the following command in command prompt - "***.\my-env\Scripts\activate***"
7. This will activate the environment. Now install all the project requirements by typing the following command in your command prompt - "***pip install -r requirements.txt***"
8. Step 7 will install the things you need to run the training and app python files.



### To Collect Data

If you want to collect only some sample data in range a to z, then you provide command line argument like this - 

```bash
python collectdata.py --range a-d
```

otherwise you can run the above command without --range argument as well.

```bash
python collectdata.py
```

The script uses your webcam to capture images. Press the corresponding  key (e.g., 'a' for 'A') to save an image in the respective directory.  Press '.' for blank images and 'Esc' to exit.

### To Train a new model 

Run the following command in your command prompt - 

```cmd
python train_model.py
```

### To test the trained model

Run the following command in your command prompt

```cmd
python app.py
```

If you have trained model, you need to update the following variable names according to your model name.
