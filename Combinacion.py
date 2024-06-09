from keras.models import Model, load_model
from keras.layers import Input, concatenate, Flatten, Dense

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

# Cargar los modelos
model_ELA = load_model("imageforgerymodelELA.keras")
model_DWT = load_model("imageforgerymodelDWT.keras")
model_CNN = load_model("imageforgerymodel.keras")

# Primera red neuronal
cnn1_output = model_ELA(input_layer)

# Segunda red neuronal
cnn2_output = model_DWT(input_layer)

# Tercera red neuronal
cnn3_output = model_CNN(input_layer)

# Concatenar las salidas de las tres redes
concatenated_output = concatenate([cnn1_output, cnn2_output, cnn3_output])

# Aplanar las características
flattened_output = Flatten()(concatenated_output)

# Capa densa para la clasificación final
dense_output = Dense(128, activation='relu')(flattened_output)
output_layer = Dense(2, activation='sigmoid')(dense_output)

# Crear el modelo combinado
combined_model = Model(inputs=input_layer, outputs=output_layer)

# Compilar el modelo
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con tus datos
history = combined_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=32)

# Evaluar el modelo
loss, accuracy = combined_model.evaluate(X_test, Y_test)
