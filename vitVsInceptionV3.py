import tensorflow as tf
import matplotlib.pyplot as plt
import os,warnings
import seaborn as sns
import tensorflow_addons as tfa
import numpy as np
import keras
import python_splitter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from pathlib import Path
from vit_keras import vit
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers,layers
from sklearn.metrics import confusion_matrix, classification_report

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)    
# else: 
#     print('GPU zaten aktif...')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

warnings.filterwarnings("ignore")

batchSize = 16
imgHeight = 128
imgWidth = 128
imgSize=(imgWidth,imgHeight)
epochs=1000


dataDir = os.path.join(str(Path(__file__).parent), "hybridDataSet")

classNamesLabels = {1 : "Meningioma",
                    2 : "Glioma",
                    3 : "Pituitary Tumor"}


####################################### Veri Seti ayarlama

if not (os.path.exists(dataDir[:-10]+"Train_Test_Folder/")) :
     python_splitter.split_from_folder(dataDir,         
                                       train=0.8,
                                       val=0.1,
                                       test=0.1)

trainDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/train")
validDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/val")
testDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/test")
chekpointDataDir = os.path.join(str(Path(__file__).parent))



trainGenerator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    samplewise_center=True,
                                    fill_mode='nearest',
                                    interpolation_order=2,
                                    samplewise_std_normalization=True,
                                    horizontal_flip=True)

testDataGenerator = ImageDataGenerator(rescale=1./255)

trainDS  = trainGenerator.flow_from_directory(
    trainDataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    shuffle = True,
    class_mode='categorical') 

valDS  = testDataGenerator.flow_from_directory(
    validDataDir ,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    class_mode='categorical')    

testDS  = testDataGenerator.flow_from_directory(
    testDataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    class_mode='categorical')    


STEP_SIZE_TRAIN = trainDS.n // trainDS.batch_size
STEP_SIZE_VALID = valDS.n // valDS.batch_size

plt.subplots(1,5, figsize=(15,15))  
for i in range(5):
        plt.subplot(1,5, i+1)
        img = next(trainDS)[0][i]
        plt.imshow(img)


callBacks = [
    tf.keras.callbacks.EarlyStopping(
                                    monitor="val_loss", 
                                    patience=15,
                                    restore_best_weights=True
    )
     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                           patience=10,
                                           factor = 0.5,
                                           min_lr = 0.00001,
                                           verbose=1
     ),
     tf.keras.callbacks.ModelCheckpoint(chekpointDataDir, 
                                         monitor='val_accuracy', 
                                         verbose=0,
                                         save_best_only=True, 
                                         save_weights_only=False,
                                         mode='auto'),
]



##################### Kendi modelim
from keras.constraints import maxnorm

hModel = Sequential()
hModel.add(Conv2D(16,(3,3),
                  padding='same',
                  activation = 'relu',
                  input_shape = ( imgHeight , imgWidth , 3),
                  kernel_constraint=maxnorm(3)))
hModel.add(MaxPooling2D())
hModel.add(Conv2D(32,(3,3),padding='same',activation = 'relu'))
hModel.add(MaxPooling2D())
hModel.add(Conv2D(64,(3,3),padding='same',activation = 'relu'))
hModel.add(MaxPooling2D())
hModel.add(Flatten())
hModel.add(BatchNormalization())
hModel.add(Dense(128, activation = tf.nn.relu))
hModel.add(Dense(64, activation = tf.nn.relu))
hModel.add(Dropout(0.1))
hModel.add(Dense(3, activation = 'softmax') )

optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001)
hModel.compile(optimizer = optimizer,
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2),
              metrics = ['accuracy'])

hModelHistory = hModel.fit(x = trainDS,
          steps_per_epoch = (STEP_SIZE_TRAIN/batchSize),
          validation_data = valDS,
          validation_steps = STEP_SIZE_VALID,
          epochs = epochs,
          callbacks = callBacks)


#####################   Vit Modeli
vit_model = vit.vit_b16(
        image_size = imgSize,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 3)

vitModel = tf.keras.Sequential([
        vit_model,
        Flatten(),
        BatchNormalization(),
        Dense(512, activation = tfa.activations.gelu),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation = tfa.activations.gelu),
        Dense(32, activation = tfa.activations.gelu),
        Dropout(0.1),
        BatchNormalization(),
        Dense(16, activation = tfa.activations.gelu),
        Dense(3, 'softmax')
    ],
    name = 'vision_transformer')

vitModel.summary()


optimizer = tfa.optimizers.RectifiedAdam(learning_rate = 0.0001)
vitModel.compile(optimizer = optimizer,
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2),
              metrics = ['accuracy'])

vitModelHistory = vitModel.fit(x = trainDS,
          steps_per_epoch = (STEP_SIZE_TRAIN/batchSize),
          validation_data = valDS,
          validation_steps = STEP_SIZE_VALID,
          epochs = epochs,
          callbacks = callBacks)
##########################  Vit Modeli




inceptModel = Sequential()
inceptionConfig= InceptionV3(include_top=False,
                            input_shape=(imgHeight, imgWidth,3),
                            pooling='avg',
                            classes=3,
                            weights='imagenet')

for layer in inceptionConfig.layers:
    layer.trainable = False   

inceptModel.add(inceptionConfig)


inceptModel.add(Flatten())
inceptModel.add(BatchNormalization())
inceptModel.add(Dense(512, activation = tfa.activations.gelu))
inceptModel.add(BatchNormalization())
inceptModel.add(Dropout(0.3))
inceptModel.add(Dense(256, activation = tfa.activations.gelu))
inceptModel.add(Dense(32, activation = tfa.activations.gelu))
inceptModel.add(Dropout(0.1))
inceptModel.add(BatchNormalization())
inceptModel.add(Dense(16, activation = tfa.activations.gelu))
inceptModel.add(Dense(3, 'softmax'))


inceptModel.summary()

optimzer=keras.optimizers.Adamax(learning_rate=0.0001);

inceptModel.compile(optimizer = optimzer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 
              metrics = ['accuracy'])



inceptModelHistory = inceptModel.fit(trainDS,
                              validation_data = valDS,
                              steps_per_epoch = (STEP_SIZE_TRAIN//batchSize),
                              # steps_per_epoch = 32,
                              epochs = epochs,
                              validation_steps = STEP_SIZE_VALID,
                              callbacks=callBacks)
    

inceptModel.save('./models/model.h5')
inceptModel.save_weights('./models/weights.h5')


predVit = np.argmax(vitModel.predict(testDS, steps = testDS.n // testDS.batch_size + 1), axis = 1)
predIncept = np.argmax(inceptModel.predict(testDS, steps = testDS.n // testDS.batch_size + 1), axis = 1)
predHmodel = np.argmax(hModel.predict(testDS, steps = testDS.n // testDS.batch_size + 1), axis = 1)
trueTable = testDS.classes
classLabels = list(testDS.class_indices.keys())  

plt.figure(figsize = (5, 5))
confMatHmodel = confusion_matrix(trueTable, predHmodel)
sns.heatmap(confMatHmodel, cmap = 'BuPu', annot = True, cbar = True)
print(confMatHmodel)
print(classification_report(trueTable, predHmodel))

plt.figure(figsize = (5, 5))
confMatVit = confusion_matrix(trueTable, predVit)
sns.heatmap(confMatVit, cmap = 'BuPu', annot = True, cbar = True)
print(confMatVit)
print(classification_report(trueTable, predVit))

confMatIncept = confusion_matrix(trueTable, predIncept)
plt.figure(figsize = (5, 5))
sns.heatmap(confMatIncept, cmap = 'BuPu', annot = True, cbar = True)
print(confMatIncept)
print(classification_report(trueTable, predIncept))



headers = ['1', '2', '3']

# ###Vit için
fig, ax = plt.subplots()
table = ax.table(cellText=confMatVit,
                  rowLabels=headers,
                  colLabels = headers,   
                  loc='center')
table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')
###Inception için
fig, ax = plt.subplots()
table = ax.table(cellText=confMatIncept,
                 rowLabels=headers,
                 colLabels = headers,   
                 loc='center')
table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')


acc = inceptModelHistory.history['accuracy']
val_acc = inceptModelHistory.history['val_accuracy']

loss = inceptModelHistory.history['loss']
val_loss = inceptModelHistory.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()













