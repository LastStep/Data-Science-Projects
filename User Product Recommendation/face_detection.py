import warnings
warnings.filterwarnings('ignore')

from PIL import Image

from numpy import asarray
from numpy import load
from numpy import expand_dims

from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

import pickle

model = pickle.loads(open('models/trained_facenet_model', "rb").read())
facenet_model = load_model('models/keras-facenet/model/facenet_keras.h5', compile=False)
data = load('trained_embeddings/celebrity-faces-embeddings.npz')
labels = data['arr_1']

# create the detector, using default weights
detector = MTCNN()
    
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def predict(embedding, image, threshold_percentage=0):
    # prediction for the face
    yhat_class = model.predict(embedding)
    yhat_prob = model.predict_proba(embedding)

    out_encoder = LabelEncoder()
    out_encoder.fit(labels)

    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
#     print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        
    return {'name':predict_names[0], 'probability':class_probability}


def detect(test_image):
    test_pixels = extract_face(test_image)

    test_embeddings =  get_embedding(facenet_model, test_pixels)
    test_embeddings = asarray(test_embeddings)

    in_encoder = Normalizer(norm='l2')
    test_embeddings = in_encoder.transform([test_embeddings])

    result = predict(test_embeddings, test_pixels)
    return result
  