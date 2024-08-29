import os
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import streamlit as st
from PIL import Image
from mtcnn import MTCNN
import pickle


detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploads', uploaded_img.name), 'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False


def extract_features(image_path,model,detector):
    img = cv2.imread(image_path)
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']
    face = img[y: y + height, x: x + width]
    #cv2.imshow('output', face)
    #cv2.waitKey(0)

    # feature extraction
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_image = np.expand_dims(face_array, axis=0)

    preprocessed_image = preprocess_input(expanded_image)
    result = model.predict(preprocessed_image).flatten()
    return result


def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title("Which Celeb You Look Like")

uploaded_img = st.file_uploader("Choose an image to be uploaded")


if uploaded_img is not None:

    # save uploaded images
    if save_uploaded_image(uploaded_img):

       display_img = Image.open(uploaded_img)

    #extract features
    features = extract_features(os.path.join('uploads', uploaded_img.name), model, detector)
    #st.text(features)
    #st.text(features.shape)

    #recommend
    index_pos = recommend(feature_list, features)
    predicted_celeb = ' '.join(filenames[index_pos].split('\\')[1].split('_'))
    #st.text(index_pos)

    col1, col2 = st.columns(2)

    with col1:
        st.header('Your uploaded image')
        st.image(display_img)
    with col2:
        st.header('Seems like ' + predicted_celeb)
        st.image(filenames[index_pos], width=300)




