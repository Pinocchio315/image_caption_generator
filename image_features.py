import os
import pickle
import tensorflow as tf

# for image-processing
from tensorflow.keras.applications import EfficientNetV2S
from keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img , img_to_array
# for modeling
from tensorflow.keras.models import Model

def extract_features(path, file_name):
    # model setup to extract features from images
    base_model = EfficientNetV2S(weights="imagenet", include_top=False)
    model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

    # feature dictionary
    features = {}
    directory = os.path.join(path, 'Images')

    for img_name in os.listdir(directory):
        img_path = directory + '/' + img_name
        # open images
        image = load_img(img_path, target_size=(224, 224))
        # preprocess
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # extract features
        feature = model.predict(image, verbose=0)
        # save
        features[img_name] = feature.reshape(-1)

    # save features in a pickle file
    pickle.dump(features, open(file_name, 'wb'))