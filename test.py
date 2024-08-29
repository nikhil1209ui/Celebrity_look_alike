from mtcnn import MTCNN
from PIL import Image
from keras_vggface.utils import preprocess_input
import numpy as np
import cv2
import pickle
from keras_vggface import VGGFace
from sklearn.metrics.pairwise import cosine_similarity

#face detection
detector = MTCNN()

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
file_names = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

sample_img = cv2.imread('samples/OIP.jpg')
result = detector.detect_faces(sample_img)

x, y, width, height = result[0]['box']
face = sample_img[y: y+height, x: x+width]
#cv2.imshow('output', face)
#cv2.waitKey(0)

#feature extraction
image = Image.fromarray(face)
image = image.resize((224, 224))
face_array = np.asarray(image).astype('float32')
expanded_image = np.expand_dims(face_array, axis=0)

preprocessed_image = preprocess_input(expanded_image)
result = model.predict(preprocessed_image).flatten()

#print(result)
#print(result.shape)

#Cosine distance
similarities = []

for i in range(len(feature_list)):
    similarities.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

index_pos = sorted(list(enumerate(similarities)), reverse=True, key=lambda x: x[1])[0][0]

temp_img = cv2.imread(file_names[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)
