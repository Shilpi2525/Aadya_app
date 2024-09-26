import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import pickle

from keras.layers import GlobalAveragePooling2D
from keras.models import Model


IMG_SIZE = (224, 224)
IMAGE_NAME = "potrait.png"

PREDICTION_RATINGS=[1.0,2.0,3.0,4.0,5.0,]
PREDICTION_RATINGS.sort()


# functions

# functions
@st.cache_resource
def get_convext_model():

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions

# get the featurization model
rating_featurized_model = get_convext_model()

# load Rating model
Rating_model = load_sklearn_models("best-rating-model")



#Building the website

#title of the web page
st.title("Celebrity Portrait Rating Webapp")

#setting the main picture
#st.image( "https://mediniz-images-2018-100.s3.ap-south-1.amazonaws.com/post-images/chokhm_1663869443.png",caption = "ABC")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.write("This web app is about give rating to the sketch")



#setting file uploader

#you can change the label name as your preference
# File uploader
image = st.file_uploader("Uplaod a pencil sketch file ", type=['jpg','jpeg','png'])

if image:

  #displaying the image
  st.image(image, caption = "User Uploaded Image")
  user_image = Image.open(image)
  # save the image to set the path
  user_image.save(IMAGE_NAME)

  #get the features
  with st.spinner("Processing......."):
   rating_image_features = featurization(IMAGE_NAME, rating_featurized_model)

    #getting prediction from rating model
    rating_model_predict = Rating_model.predict(rating_image_features)

    rating_model_predict_proba = Rating_model.predict_proba(rating_image_features)
    rating = PREDICTION_RATINGS[int(rating_model_predict)]
    rating_probability = rating_model_predict_proba[0][int(rating_model_predict[0])] 

    st.header("Portrait Rating")
    st.header("{}".format(PREDICTION_RATINGS[int(rating_model_predict[0])]))
