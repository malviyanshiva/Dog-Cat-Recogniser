
from flask import Flask, render_template, request
import os

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


filepath ='model.h5'


model = load_model(filepath)
print(model)

print("Model Loaded Successfully")




def pred(image):
  test_image = load_img(image, target_size = (64,64)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255.0 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  pred = model.predict(test_image) 
  print('@@ Raw result = ', pred)
  pred=np.around(pred)
  print(int(pred[0][0]))
  if pred[0][0]==1:
      return "Dog"
       
  else:
      return "Cat"
        
 

# Create flask instance
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html',background_image="static/background.jpg")

@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename   
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('./static/upload/', "upload.jpg")
        file.save(file_path)
        print("@@ Predicting class......")
        prediction = pred(image=file_path)
        print(file_path)
        return render_template('pred.html', pred_output = prediction, user_image = file_path,background_image="static/predictionpage.jpeg")



if __name__ == "__main__":
    app.run(debug=True)

