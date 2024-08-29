import numpy as np
import tensorflow as tf
import tqdm
from keras_preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import os
import pickle

#actors = os.listdir('data')
#print(actors)

#filenames = []
#for actor in actors:
#    for file in os.listdir(os.path.join('data', actor)):
#       filenames.append(os.path.join('data', actor, file))


#pickle.dump(filenames, open('filenames.pkl', 'wb'))

filenames = pickle.load(open('filenames.pkl', 'rb'))
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#model.summary()


def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expended_img = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expended_img)
    result = model.predict(preprocessed_img).flatten()

    return result


features = []
for file in tqdm.tqdm(filenames):
    result = feature_extractor(file, model)
    features.append(result)

pickle.dump(features, open('embeddings.pkl', 'wb'))

