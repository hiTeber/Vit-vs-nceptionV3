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
from sklearn.decomposition import PCA
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


warnings.filterwarnings("ignore")

batchSize = 8
imgHeight = 224
imgWidth = 224
imgSize=(imgWidth,imgHeight)
epochs=300


# dataDir = os.path.join(str(Path(__file__).parent), "brainTumor")
dataDir = os.path.join(str(Path(__file__).parent), "hybridDataSet")



classNamesLabels = {1 : "Meningioma",
                    2 : "Glioma",
                    3 : "Pituitary Tumor"}



# if not (os.path.exists(dataDir[:-10]+"Train_Test_Folder/")) :
#     python_splitter.split_from_folder(dataDir,         
#                                       train=0.7,
#                                       val=0.15,
#                                       test=0.15)

trainDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/train")
validDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/val")
testDataDir = os.path.join(str(Path(__file__).parent), "Train_Test_Folder/test")
chekpointDataDir = os.path.join(str(Path(__file__).parent))

def kMean(image):
    pca = PCA()
    transformed_image = pca.inverse_transform(pca.fit_transform(image.reshape(-1, 1)))
    clf = KMeans(n_clusters=3, n_init='auto')
    clf.fit(transformed_image)
    pixels_seg0 = clf.cluster_centers_[clf.labels_]
    pixels_seg = np.clip(pixels_seg0, 0, 255)
    image_seg = pixels_seg.reshape(image.shape)
    return image_seg

trainGenerator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    samplewise_center=True,
                                    fill_mode='nearest',
                                    interpolation_order=2,
                                    # preprocessing_function=kMean,
                                    samplewise_std_normalization=True,
                                    horizontal_flip=True)

testDataGenerator = ImageDataGenerator(rescale=1./255)

trainDS  = trainGenerator.flow_from_directory(
    trainDataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    shuffle = True,
    seed = 333,
    class_mode='categorical') 

valDS  = testDataGenerator.flow_from_directory(
    validDataDir ,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    seed = 333,
    class_mode='categorical')    

testDS  = testDataGenerator.flow_from_directory(
    testDataDir,
    target_size=(imgHeight, imgWidth),
    batch_size = batchSize,
    color_mode = 'rgb',
    seed = 333,
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
                                    patience=10,
                                    restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", 
                                          patience=10,
                                          factor = 0.5,
                                          min_lr = 0.00001,
                                          verbose=1
    ),
    # tf.keras.callbacks.ModelCheckpoint(chekpointDataDir, 
    #                                     monitor='val_accuracy', 
    #                                     verbose=0,
    #                                     save_best_only=True, 
    #                                     save_weights_only=False,
    #                                     mode='auto'),
]
optimizer = tf.optimizers.Adam(learning_rate = 0.0001)



#####################   Vit Modeli
vit_model = vit.vit_b16(
        image_size = imgSize,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 5)

vitModel = tf.keras.Sequential([
        vit_model,
        Flatten(),
        BatchNormalization(),
        Dense(512, activation = tfa.activations.gelu),
        Dense(128, activation = tfa.activations.gelu),
        Dense(64, activation = tfa.activations.gelu),
        Dense(16, activation = tfa.activations.gelu),
        Dense(3, 'softmax')
    ],
    name = 'vision_transformer')


vitModel.summary()



vitModel.compile(optimizer = optimizer,
              loss = tf.keras.losses.CategoricalCrossentropy(),
              metrics = ['accuracy'])
with tf.device('/gpu:0'):
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
                            weights='imagenet')

for layer in inceptionConfig.layers:
    layer.trainable = False   

inceptModel.add(inceptionConfig)


inceptModel.add(Flatten())
inceptModel.add(BatchNormalization())
inceptModel.add(Dense(512, activation = tfa.activations.gelu))
inceptModel.add(Dense(128, activation = tfa.activations.gelu))
inceptModel.add(Dense(64, activation = tfa.activations.gelu))
inceptModel.add(Dense(16, activation = tfa.activations.gelu))
inceptModel.add(Dense(3, 'softmax'))


inceptModel.summary()


inceptModel.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 
              metrics = ['accuracy'])


with tf.device('/gpu:0'):
    inceptModelHistory = inceptModel.fit(trainDS,
                                  validation_data = valDS,
                                  steps_per_epoch = (STEP_SIZE_TRAIN//batchSize),
                                  epochs = epochs,
                                  validation_steps = STEP_SIZE_VALID,
                                  callbacks=callBacks)
    

inceptModel.save('./models/model.h5')
inceptModel.save_weights('./models/weights.h5')

vitModel.save('./models/model.h5')
vitModel.save_weights('./models/weights.h5')


predVit = np.argmax(vitModel.predict(testDS, steps = testDS.n // testDS.batch_size + 1), axis = 1)
predIncept = np.argmax(inceptModel.predict(testDS, steps = testDS.n // testDS.batch_size + 1), axis = 1)
trueTable = testDS.classes
classLabels = list(testDS.class_indices.keys())  


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
ax.set_title('Vit Model')
ax.axis('off')
###Inception için
fig, ax = plt.subplots()
table = ax.table(cellText=confMatIncept,
                 rowLabels=headers,
                 colLabels = headers,
                 loc='center')
table.set_fontsize(14)
table.scale(1,4)
ax.set_title('InceptionV3')
ax.axis('off')


acc = inceptModelHistory.history['accuracy']
val_acc = inceptModelHistory.history['val_accuracy']
inceptionModelAcc = sum(acc)/len(acc)
inceptionModelValAcc = sum(val_acc)/len(val_acc)
print("İnceptionV3 Acc " + str(inceptionModelAcc))
print("İnceptionV3 Val_Acc " +  str(inceptionModelValAcc))

accVit = vitModelHistory.history['accuracy']
valAccVit = vitModelHistory.history['val_accuracy']
vitModelAcc = sum(accVit)/len(accVit)
vitModelValAcc = sum(valAccVit)/len(valAccVit)
print("Vit Acc " +  str(vitModelAcc))
print("Vit Val_Acc " +  str(vitModelValAcc))

loss = inceptModelHistory.history['loss']
val_loss = inceptModelHistory.history['val_loss']
inceptionModLoss = sum(loss)/len(loss)
inceptionModValLoss = sum(val_loss)/len(val_loss)
print("İnceptionV3 loss " +  str(inceptionModLoss))
print("İnceptionV3 Val_loss " +  str(inceptionModValLoss))

lossVit = vitModelHistory.history['loss']
val_lossVit = vitModelHistory.history['val_loss']
vitModelLoss = sum(lossVit)/len(lossVit)
vitModelValLoss = sum(val_lossVit)/len(val_lossVit)
print("Vit Loss " +  str(vitModelLoss))
print("Vit Val_loss " +  str(vitModelValLoss))

plt.figure(figsize=(8, 8))
plt.subplot(4, 2, 1)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('İnceptionV3 Eğitim ve Doğrulama Doğruluğu')

plt.subplot(4, 2, 2)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.5])
plt.title('İnceptionV3 Eğtim ve Doğrulama Kaybı')
plt.xlabel('epoch')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(4, 2, 3)
plt.plot(accVit, label='Training')
plt.plot(valAccVit, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('Vit Eğitim ve Doğrulama Doğruluğu')

plt.subplot(4, 2, 4)
plt.plot(lossVit, label='Training')
plt.plot(val_lossVit, label='Validation')
plt.legend(loc='lower right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.5])
plt.title('Vit Eğitim ve Doğrulama Kaybı')
plt.xlabel('epoch')
plt.show()






