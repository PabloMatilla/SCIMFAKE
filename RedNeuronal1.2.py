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
from skimage import io, color, transform, restoration, filters
import pywt
import cv2

global n
n = 0
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

def mostrar_imagenes(img1, img2):
    # Crear subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Mostrar la primera imagen en el primer subgráfico
    axes[0].imshow(img1)
    axes[0].axis('off')  # Desactivar los ejes
    axes[0].set_title('Imagen original')

    # Mostrar la segunda imagen en el segundo subgráfico
    axes[1].imshow(img2)
    axes[1].axis('off')  # Desactivar los ejes
    axes[1].set_title('Imagen procesada')

    # Mostrar los subgráficos
    plt.show()


def open_image(path):
    image = Image.open(path).convert('RGB')
    return image

def preprocessing(image_path, target_size=(128, 128)):
    # Cargar la imagen
    image = img = open_image(image_path)
    image = image.resize(target_size)

    # Convertir la imagen a un array NumPy y normalizar los valores de píxeles
    processed_image = np.array(image).flatten() / 255.0
    # mostrar_imagenes(img, image)

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


def CNN():
    model = Sequential()

    model.add(Input(shape=(128, 128, 1))) # Cambiamos el shape (128, 128, 3) --> (128, 128)

    # Cinco capas convolucionales con normalización por lotes y agrupamiento máximo
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Aplanar las características antes de pasarlas a las capas densas
    model.add(Flatten())

    # Cuatro capas densas con normalización por lotes
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())

    # Capa de salida con activación softmax para la clasificación binaria
    model.add(Dense(2, activation='sigmoid'))

    return model


# Crear una instancia del modelo personalizado
model = CNN()

# Compilar el modelo con el optimizador Adam y la función de pérdida Binary Cross Entropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Imprimir un resumen del modelo
model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=200, batch_size=32)
loss, accuracy = model.evaluate(X_test, Y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("accuracy (DWT)")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Validation'])
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss (DWT)")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train','Validation'])
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix (DWT)',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(Y_test, axis = 1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(2))

model.save("imageforgerymodelDWT.keras")

laodedmodel = load_model("imageforgerymodelDWT.keras")
laodedmodel.summary()