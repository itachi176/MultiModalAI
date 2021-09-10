import os 
import cv2
import numpy as np 
import mtcnn 
from imutils import paths
from  save_file import *
import matplotlib.pyplot as plt

path = './pretrain_model'
a=  os.path.join(path, "nn4.small2.v1.t7")
def load_torch(path):
    model = cv2.dnn.readNetFromTorch(path)
    return model
encoder = load_torch(a)
#doc anh 
def image_read(image_path):
  """
  input:
    image_path: link file ảnh
  return:
    image: numpy array của ảnh
  """
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  return image

#blob image 
def _blobImage(image, out_size = (300, 300), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0)):
  """
  input:
    image: ma trận RGB của ảnh input
    out_size: kích thước ảnh blob
  return:
    imageBlob: ảnh blob
  """
  # Chuyển sang blobImage để tránh ảnh bị nhiễu sáng
  imageBlob = cv2.dnn.blobFromImage(image, 
                                    scalefactor=scaleFactor,   # Scale image
                                    size=out_size,  # Output shape
                                    mean=mean,  # Trung bình kênh theo RGB
                                    swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                    crop=False)
  return imageBlob

blob =  _blobImage(image=cv2.cvtColor(cv2.imread("face.jpg"), cv2.COLOR_BGR2RGB), out_size = (300, 300), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0))

# extract face
def extract_face(image, thresh_scale = (20, 20)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detect = mtcnn.MTCNN()
    try:
      bbox = detect.detect_faces(image)[0]['box']
    except:
      return None
    # faces = []
    if bbox[2] > 20 or bbox[3] > 20: 
        img = image[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]
        face = img.copy()

    else:
      face = None
    return face 
# img = extract_face(cv2.imread('test.jpg'))
# print(img)
#processing

datasets_path = './dataset/'
# def process(thresh_scale= (20, 20)):
#     image_links = list(paths.list_images(datasets_path))
#     images_file=[]
#     y_labels = []
#     faces = []
#     total = 0
#     for image_link in image_links:
#       #lay nhan 
#       split_link = image_link.split("/")
#       split_link2 = split_link[2].split("\\")
#       label = split_link2[0]
#       img = image_read(image_link)
#       (h, w) = img.shape[:2]
#       face = extract_face(img, thresh_scale=(20, 20))
#       if face is not None:
#         faces.append(face)
#         y_labels.append(label)
#         images_file.append(image_link)
#         total += 1
#       else:
#         next
#       print("processed -------- "+ str(total))
#     return faces, y_labels, images_file

# faces, y_labels, images_file = process(thresh_scale=(20, 20))


# save(faces, "./model/faces.pkl")
# save(y_labels, "./model/y_labels.pkl")
# save(images_file, "./model/images_file.pkl")
faces = load("./model/faces.pkl")
y_labels = load("./model/y_labels.pkl")
images_file = load("./model/images_file.pkl")

def _embedding_faces(encoder, faces):
  emb_vecs = []
  for face in faces:
    faceBlob = _blobImage(face, out_size = (96, 96), scaleFactor=1/255.0, mean=(0, 0, 0))
    # Embedding face
    encoder.setInput(faceBlob)
    vec = encoder.forward()
    emb_vecs.append(vec)
  return emb_vecs

embed_faces = _embedding_faces(encoder, faces)
# Nhớ save embed_faces vào Dataset.
save(embed_faces, "./model/embed_blob_faces.pkl")
embed_faces = load("./model/embed_blob_faces.pkl")
y_labels = load("./model/y_labels.pkl")

from sklearn.model_selection import train_test_split
ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids, test_size = 0.2, stratify = y_labels)
X_train = np.squeeze(X_train, axis = 1)
X_test = np.squeeze(X_test, axis = 1)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))
save(id_train, "./model/id_train.pkl")
save(id_test, "./model/id_test.pkl")

from sklearn.metrics.pairwise import cosine_similarity

def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

# Lấy ngẫu nhiên một bức ảnh trong test
vec = X_test[1].reshape(1, -1)
# Tìm kiếm ảnh gần nhất
print(_most_similarity(X_train, vec, y_train))

#def _acc_test(test_set, y_test):
from sklearn.metrics import accuracy_score

y_preds = []
for vec in X_test:
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(X_train, vec, y_train)
  y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))

#base model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

def _base_network():
  model = VGG16(include_top = True, weights = None)
  dense = Dense(128)(model.layers[-4].output)
  norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(dense)
  model = Model(inputs = [model.input], outputs = [norm2])
  return model

model = _base_network()


faces = load("./model/faces.pkl")

faceResizes = []
for face in faces:
  face_rz = cv2.resize(face, (224, 224))
  faceResizes.append(face_rz)
X = np.stack(faceResizes)
#split data 
id_train = load("./model/id_train.pkl")
id_test = load("./model/id_test.pkl")

X_train, X_test = X[id_train], X[id_test]

#loss 
import tensorflow_addons as tfa

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tfa.losses.TripletSemiHardLoss())

# #train 
# gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(32)

# history = model.fit(
#     gen_train,
#     steps_per_epoch = 50,
#     epochs=10)

# # save model 
# model.save("model/model_triplot.h5")

X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)
y_preds = []
for vec in X_test_vec:
  vec = vec.reshape(1, -1)
  y_pred = _most_similarity(X_train_vec, vec, y_train)
  y_preds.append(y_pred)

print("acc", accuracy_score(y_preds, y_test))

def normalize_image(image, epsilon=0.000001):
  means = np.mean(image.reshape(-1, 3), axis=0)
  stds = np.std(image.reshape(-1, 3), axis=0)
  image_norm = image - means
  image_norm = image_norm/(stds + epsilon)
  return image_norm

IMAGE_OUTPUT = "./test1.jpg"
IMAGE_PREDICT = "./messi.jpg"

# Trích xuất bbox image 
image = image_read(IMAGE_PREDICT)
# imageBlob = _blobImage(image)
# print(len(bboxs))
faces = []
face = extract_face(image, thresh_scale= (20, 20))
# face = face.copy()
faces.append(face)
try:
  face_rz = cv2.resize(face, (224, 224))
  # Chuẩn hóa ảnh bằng hàm _normalize_image
  face_tf = normalize_image(face_rz)
  face_tf = np.expand_dims(face_tf, axis = 0)
  # Embedding face
  vec = model.predict(face_tf)
  # Tìm kiếm ảnh gần nhất
  name = _most_similarity(X_train_vec, vec, y_train)
  # Tìm kiếm các bbox
  
  print(name)
except:
    print("Not found face")
cv2.imwrite(IMAGE_OUTPUT, image)


# import matplotlib.pyplot as plt

plt.figure(figsize = (16, 8))
img = plt.imread(IMAGE_OUTPUT)
plt.imshow(img)



