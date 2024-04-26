### Dataset

1. Download the dataset from [here](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
2. Extract and put the dataset in data folder. It will look like
   ```
   data
     - train
     - validation
   ```
   
### Environment
1. Install TensorFlow from [here](https://www.tensorflow.org/install/pip). I am using Linux so I used `python3 -m pip install tensorflow[and-cuda]`
2. Install TensorFlow JS using `pip install tensorflowjs`

### How to run the code?
1. Train the model: `python train_model.py`
2. Convert the `.h5` model to TensorFlow JS model using: `tensorflowjs_converter --input_format keras saved_model/tf_model.h5 saved_model/tfjs_model/`
3. Start the application using: `python3 -m http.server`
