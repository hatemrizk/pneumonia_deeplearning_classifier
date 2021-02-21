import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, jsonify, make_response
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.vgg16 import decode_predictions

METRICS = [
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),

]

UPLOAD_FOLDER = 'upload'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

image_size=224

base_model = tf.keras.applications.InceptionResNetV2(include_top= False, 
                                          input_shape=(image_size,image_size,3),
                                          
                                          weights='imagenet')
                                          
def make_model(base_model, num_units=256, dropout=0.2, learning_rate=0.0001, optimizer='adam', metrics=METRICS):

# This helper function creates and compiles out deep learning model.
# num_units: number of units in Dense layers.
# dropout: a fraction representing percentage of activation nodes to drop out.
# learning_rate: optimizer learning rate.
# optimizer: string representing optimzer module. {'adam', 'sgd, 'rmsprop'}
# metrics: Keras metrics


  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D(input_shape=(2048,1,1))(x)
  x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dropout(dropout)(x)
  x = tf.keras.layers.Dense(num_units, activation='relu')(x)
  #x = BatchNormalization()(x)
  x = tf.keras.layers.Dropout(dropout)(x)


  predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  for layer in base_model.layers:
    layer.trainable = False
  
  if optimizer== 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer== 'sgd': 
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer== 'rmsprop': 
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
  else:
    raise ValueError('Unexpected optimizer name: %r' % optimizer)

  #Compile model

  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(), 
                metrics=metrics)

  return model

MODEL_WEIGHTS_PATH = 'models/mdl_incp_rsnt_weights.h5'

model = make_model(base_model,  256, 0.2,  0.001,'adam', METRICS)

model.summary()
model.load_weights(MODEL_WEIGHTS_PATH)

#model.save('models/mdl_incp_rsnt_lcl.h5')

#model = load_model(MODEL_PATH)
model.make_predict_function()

print("Model loaded. Start serving..")
print("Model loaded. Check http://0.0.0.0:8080/")

def model_predict(img_path, model):
   img = image.load_img(img_path, target_size=(224, 224))
   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   x = preprocess_input(x)
   preds = model.predict(x)
   return preds

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		file_name = secure_filename(f.filename)
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
		f.save(file_path)
		
		preds = model_predict(file_path, model)
		#pred_class = decode_predictions(preds, top=1)
		#result = str(pred_class[0][0][1])
		result = 'Pneumonia Detected' if preds[0] > 0.5 else 'Pneumonia Not Detected'
	return result
	
	
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=False)

