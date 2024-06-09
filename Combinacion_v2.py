from keras.models import Model, load_model
from keras.layers import Input, concatenate, Flatten, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import itertools
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from keras.layers import Dense, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

total_files = glob("casia/CASIA2/**/*")
print(len(total_files))

type=set()
for file in total_files:
    type.add(file.split(".")[-1])
print("types of files are",type)

jpg_images=glob("casia/CASIA2/**/*.jpg")
print('the number of jpg images are:',len(jpg_images))

tif_images=glob("casia/CASIA2/**/*.tif")
print("the number of tif images are:",len(tif_images))

bmp_images=glob("casia/CASIA2/**/*.bmp")
print("the number of bmp images are:",len(bmp_images))


images=jpg_images
print('jpeg files',len(images))
print('tif_files',len(tif_images))

def findfiles(files):
    tp_files,au_files=0, 0
    for file in files:
        if file.split("/")[-2]=="Tp":
            tp_files+=1
        if file.split("/")[-2]=="Au":
            au_files+=1
    return tp_files,au_files

print(findfiles(images))


def open_image(path):
    image = Image.open(path).convert('RGB')
    return image


def preprocessing(image_path, target_size=(128, 128)):
    # Cargar la imagen y redimensionarla
    image = open_image(image_path)
    image_resized = image.resize(target_size)

    # Convertir la imagen a un array NumPy y normalizar los valores de píxeles
    processed_image = np.array(image_resized).flatten() / 255.0

    return processed_image


## CARGA DE BASES DE DATOS

# CASIA2
X=[]
Y=[]
path= "casia/CASIA2/Au"
cant = 0
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
            if cant == 6044: break
            full_path = os.path.join(dirname, filename)
            X.append(preprocessing(full_path))
            Y.append(0)
            cant += 1

print("Tamaño de Au de CASIA2: ", len(X))

v=0
path = "casia/CASIA2/Tp"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            X.append(preprocessing(full_path))
            Y.append(1)
            v+=1

print("Tamaño de Tp de CASIA2: ", v)
print("Tamaño de Au + Tp CASIA2: ", len(X))

# 218 imagenes manipuladas de CASIA1
n = 0
path = "casia/CASIA1/Sp"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('tif'):
            #if n == 217: break
            full_path = os.path.join(dirname, filename)
            X.append(preprocessing(full_path))
            Y.append(1)
            n += 1

print("Tamaño de Tp de CASIA2: ", n)


print("# imagenes manipuladas (Tp): ", Y.count(1))
print("# imagnes sin manipular (Au): ", Y.count(0))

X = np.array(X)
X = X.reshape(-1, 128, 128)

Y = to_categorical(Y, 2)

X_train_test,  X_val,  Y_train_test,  Y_val  = train_test_split(X,  Y,  test_size=0.2, random_state=10)
X_train,  X_test, Y_train,  Y_test  =  train_test_split(X_train_test, Y_train_test, test_size=0.25, random_state=2)

print("# set entrenamiento: ", len(X_train))
print("# set de text: ", len(X_test))
print("# set de validación: ", len(X_val))


# Define las tres redes neuronales
input_shape = (128, 128, 3)  # Ajusta esto según el tamaño de tus imágenes
input_layer = Input(shape=input_shape)



# Cargar el modelo
loaded_model_1 = load_model("imageforgerymodel.keras")

# Crear un nuevo modelo que termina en la penúltima capa densa
feature_extractor_1 = Model(inputs=loaded_model_1.input, outputs=loaded_model_1.layers[-2].output)

# Extraer características para el conjunto de entrenamiento y prueba
features_train_1 = feature_extractor_1.predict(X_train)
features_test_1 = feature_extractor_1.predict(X_test)

# Cargar el modelo ELA
loaded_model_ELA = load_model("imageforgerymodelELA.keras")

# Crear un nuevo modelo que termina en la penúltima capa densa
feature_extractor_ELA = Model(inputs=loaded_model_ELA.input, outputs=loaded_model_ELA.layers[-2].output)

# Extraer características para el conjunto de entrenamiento y prueba
features_train_ELA = feature_extractor_ELA.predict(X_train)
features_test_ELA = feature_extractor_ELA.predict(X_test)


# Cargar el modelo DWT
loaded_model_DWT = load_model("imageforgerymodelDWT.keras")

# Crear un nuevo modelo que termina en la penúltima capa densa
feature_extractor_DWT = Model(inputs=loaded_model_DWT.input, outputs=loaded_model_DWT.layers[-2].output)

# Extraer características para el conjunto de entrenamiento y prueba
features_train_DWT = feature_extractor_DWT.predict(X_train)
features_test_DWT = feature_extractor_DWT.predict(X_test)


# Supongamos que tienes características de tres modelos diferentes
# features_train_2, features_train_3, features_test_2, features_test_3

# Combina las características
combined_features_train = np.hstack((features_train_1, features_train_ELA, features_train_DWT))
combined_features_test = np.hstack((features_test_1, features_test_ELA, features_test_DWT))

# Entrena el clasificador meta
meta_classifier = LogisticRegression()
meta_classifier.fit(combined_features_train, np.argmax(Y_train, axis=1))

# Predice con el clasificador meta
y_pred = meta_classifier.predict(combined_features_test)

# Evalúa la precisión
accuracy = accuracy_score(np.argmax(Y_test, axis=1), y_pred)
print(f"Meta Classifier Accuracy: {accuracy}")
