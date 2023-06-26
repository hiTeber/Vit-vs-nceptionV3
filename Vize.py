import numpy as np
import pandas as pd  #Verileri manipulasyonu için Pandas kullanılır 
from keras.layers.core import Dense,Activation,Dropout,Flatten,Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split # verileri karıştırmak ve 2ye bölmek için kullanılan kütüphane
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score,KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import pyplot
np.set_printoptions(threshold=np.inf)



hepatitData = pd.read_csv('HepatitisCdata.csv',index_col="Unnamed: 0") 
#Hepatit verisini okuduk pandas ile. İlk sutunu attık

# Veri Ön işleme aşaması 
# values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis'
colHepatitData= ['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
outputHepData= ['Blood Donor','Suspect Blood Donor','Hepatit','Fibrosis','Cirrhosis']

tempVal=[]
for x in hepatitData['Category']:  #kategori bölmesindeki verilerde yazan string ifadeleri temizledik
    if x[:2]=='0s':
        tempVal.append(-1)  #şüpheli kan vericisini 0s den -1 konumuna getirdik.
    elif x[:2]!='0s':
        tempVal.append(x[:1])
        
hepatitData['Category']=[int(i) for i in  tempVal]  #Veri dosyasında category de yazı ifadeleri vardı onları çıkardık sadece 0,1,2,3 olacak şekilde düzenledik.

#cinsiyet verisini erkek -1, kadın -2 olacak şekilde düzenledik 
tempSexVal=[]
for x in hepatitData['Sex']:
    if x =='m':
        tempSexVal.append(-1)
    if x == 'f':
        tempSexVal.append(-2);
hepatitData['Sex']=[int(i) for i in tempSexVal]; 

#Nan değerleri ortalamaya göre doldurma
hepatitData["ALB"].fillna(hepatitData['ALB'].mean(), inplace = True)
hepatitData["ALP"].fillna(hepatitData['ALP'].mean(), inplace = True)
hepatitData["ALT"].fillna(hepatitData['ALT'].mean(), inplace = True)
hepatitData["CHOL"].fillna(hepatitData['CHOL'].mean(), inplace = True)
hepatitData["PROT"].fillna(hepatitData['PROT'].mean(), inplace = True)

#normalize
norHepatitData = hepatitData.drop(columns=['Category'])
norHepatitData = preprocessing.minmax_scale(norHepatitData)


#korelasyon analizi
hepatitDataCorr = hepatitData.corr().abs() #korelasyon matrisi oluşturma
#hepatitDataCorr = preprocessing.minmax_scale(hepatitDataCorr)


plt.figure(figsize=(10,10), dpi=500) #figürün boyutunu ayarladığımız bölüm
sns.heatmap(hepatitDataCorr,cmap='coolwarm', annot=True, linewidths=1) 
#ısı haritası şeklinde özniteliklerin korelasyonunu gösteriyor


#category bölümü hastalık türleri olduğu için bu bölüm çıkış olacak 
#Bu stun haricindeki bölümler giriş olacak ve eğitilecek train_x eğitilecek verileri aktarıyoruz.
#train_X = hepatitData.drop(columns=['Category'])
train_X = norHepatitData
train_Y = hepatitData[['Category']] #çıkış verilerini Y ye aktardık.
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.15, shuffle=True, random_state=100)
Y_train=Y_train.values.ravel()
#yukarıda verileri 2 parçaya ayırdık.Bunun %85Eğitim verisi %15 ise test verisi olarak ayarlandı.
X_train70, X_test70, Y_train70, Y_test70 = train_test_split(train_X, train_Y, test_size=0.30, shuffle=True, random_state=100)
Y_train70=Y_train70.values.ravel()
#yukarda veri setini %70 Eğitim verisi olarak ayarladık

models = [
    {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(),
        "hyperparameters": {
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "max_iter": [50,100,200]
        }
    },
    {
     "name":'KNN algoritması',
     "estimator": KNeighborsClassifier(),
     "hyperparameters":{
         "n_neighbors" : [1,3,5,7],
         "weights":["uniform"]
         }
     
     },
    {
     "name":'Naive bayes',
     "estimator": GaussianNB(),
     "hyperparameters":{
         "priors":[None],
         "var_smoothing":[1e-09,1e-06]
         }
     },
    {
     "name":'Karar Ağacı',
     "estimator": DecisionTreeClassifier(),
     "hyperparameters":{
         "random_state":[0,2,3],
         "max_depth":[1,2,4,10,20]
         }
     },
    {
     "name":'Çok Katmanlı',
     "estimator": MLPClassifier(),
     "hyperparameters":{
         "hidden_layer_sizes":[50,100,200],
         "activation":['relu'],
         "solver":['adam'],
         "max_iter":[50,100,300],
         "learning_rate":['constant'],
         "learning_rate_init":[0.1,0.01,0.001]
         
         }
     },
    {
        "name": "Rasgele Ağaç",
        "estimator": RandomForestClassifier(),
        "hyperparameters": {
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 10, 20, None]
        }
    },
    {
        "name": "Destek Vektör Makinesi",
        "estimator": SVC(),
        "hyperparameters": {
            "C": [0.01, 0.1, 1, 10],
            "kernel": ["linear", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"]
        }
    }
]

kerasModel = Sequential() #keras da sequential modeli oluşturdul
kerasModel.add(Dense(60, input_dim=12,kernel_initializer='uniform',activation='relu')) #modelimize katman ekliyoruz ilk katmanda bu değerleri yazmamız gerekli
kerasModel.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))    
kerasModel.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])                
kerasModel.fit(X_train, Y_train , batch_size=8, epochs=250)

history = kerasModel.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=250, verbose=0)
train_mse = kerasModel.evaluate(X_train, Y_train, verbose=0)
test_mse = kerasModel.evaluate(X_test, Y_test, verbose=0)
y_keras_pred = kerasModel.predict(X_test)

pyplot.figure(figsize=(10,10),dpi=500)
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.ylim(0, 1)
pyplot.legend()
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
kerasModel.summary()
pyplot.ylim(0, 1)
pyplot.show()
accuracies = []
precisionData=[]
AUC=[1,2,3,4,5,6,7]
F1scoreData = []
recallData=[]
bestModel = {}

for model in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
       
        gridSearch = GridSearchCV(
            estimator=model['estimator'],
            param_grid=model['hyperparameters'],
            scoring='accuracy',
            cv=5
        )
        gridSearch.fit(X_train, Y_train)
        best_model = gridSearch.best_estimator_
        y_pred = best_model.predict(X_test)
        #AUC.append(roc_auc_score(Y_test, y_pred,multi_class='ovo'))
        accuracy = accuracy_score(Y_test, y_pred)
        precisionData.append(precision_score(Y_test, y_pred,average='weighted'))
        recallData.append(recall_score(Y_test, y_pred,average='weighted'))
        F1scoreData.append(f1_score(Y_test, y_pred,average='weighted'))
        accuracies.append((model['name'], accuracy))
        bestModel[model['name']] = bestModel
        print(f"\nEn iyi parametreler %85 Veri -> {model['name']}: {gridSearch.best_params_}")
        print(f"\nDoğruluk -> {model['name']}: {accuracy}")
        print("----------------------------------------------------------------------")
        
accuracies70 = []
precisionData70=[]
AUC70=[1,2,3,4,5,6,7]
F1scoreData70 = []
recallData70=[]

for model in models:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
       
        gridSearch = GridSearchCV(
            estimator=model['estimator'],
            param_grid=model['hyperparameters'],
            scoring='accuracy',
            cv=5
        )
        gridSearch.fit(X_train70, Y_train70)
        best_model = gridSearch.best_estimator_
        y_pred = best_model.predict(X_test70)
        
        accuracy = accuracy_score(Y_test70, y_pred)
        precisionData70.append(precision_score(Y_test70, y_pred,average='weighted'))
        recallData70.append(recall_score(Y_test70, y_pred,average='weighted'))
        F1scoreData70.append(f1_score(Y_test70, y_pred,average='weighted'))
        # AUC70.append(roc_auc_score(Y_test, y_pred,multi_class='ovr'),average='micro')
        accuracies70.append((model['name'], accuracy)) 
        bestModel[model['name']] = bestModel
        print(f"\nEn iyi parametreler %70 Veri -> {model['name']}: {gridSearch.best_params_}")
        print(f"\nDoğruluk -> {model['name']}: {accuracy}")
        print("----------------------------------------------------------------------")
 
#lojik regresyon sınıflandırması
logicRegresModel = LogisticRegression(C=10, max_iter=200,penalty='l2')
logicRegresModel.fit(X_train, Y_train)  #eğitim verileri üzerinde eğitim yaptığımız bölüm
y_pred_log_reg = logicRegresModel.predict(X_test) #test verileri üzerinde tahmin yaptığımız bölüm

#KNn sınıflandırması
KNNmodel = KNeighborsClassifier(n_neighbors=1,weights='uniform')
KNNmodel.fit(X_train,Y_train)
y_pred_knn = KNNmodel.predict(X_test)

#Rasgele Ağaç sınıflandırması
forestModel = RandomForestClassifier(n_estimators=200)
forestModel.fit(X_train, Y_train)
y_pred_rforest = forestModel.predict(X_test)

#Destek Vektör Makine sınıflandırması
supVectMacModel = SVC(C=10,gamma='scale',kernel='rbf')
supVectMacModel.fit(X_train, Y_train) 
y_pred_svm = supVectMacModel.predict(X_test)

#karar ağacı sınıflandırma
decisionTreeModel = DecisionTreeClassifier(random_state=0, max_depth=4)
decisionTreeModel = decisionTreeModel.fit(X_train,Y_train)
y_pred_decisTree = supVectMacModel.predict(X_test)

#naive_bayes 
naiveGaus = GaussianNB(var_smoothing = 1e-09)
naiveGaus.fit(X_train, Y_train)
y_pred_naiveGaus = naiveGaus.predict(X_test)

#Çok katmanlı Sınıflandırıcı
mlpClass=MLPClassifier(batch_size=50,
                       hidden_layer_sizes=200,
                       activation='relu',
                       solver='adam',
                       learning_rate_init=0.001,
                       max_iter=100)
mlpClass.fit(X_train, Y_train)
y_pred_mlpClass = mlpClass.predict(X_test)

predList = {
    'Lojistik Regresyon': y_pred_log_reg,
    'KNN algoritması' : y_pred_knn,
    'Rasgele Ağaç' : y_pred_rforest,
    'Destek Vektör Makineleri' : y_pred_svm,
    'Karar Ağacı' : y_pred_decisTree,
    'Naive bayes' : y_pred_naiveGaus,
    'Çok Katmanlı' : y_pred_mlpClass,
    'Keras': y_keras_pred
    }
modelsObjects = {
    'Lojistik Regresyon': logicRegresModel,
    'KNN algoritması' : KNNmodel,
    'Rasgele Ağaç': forestModel,
    'Destek Vektör Makineleri': supVectMacModel,
    'Naive bayes': naiveGaus,
    'Karar Ağacı': decisionTreeModel,
    'Çok Katmanlı':mlpClass,
}

#Cross-Validation değerlendirmesi
modelObjNames =[]
modelObjValues = []
crosValScore = {}
kf=KFold(n_splits=5)
tempVal=[]
for name,val in modelsObjects.items():
    modelObjNames.append(name)
    dicVal = cross_val_score(val,X_train,Y_train,cv=kf).mean()
    tempVal.append(dicVal)
    crosValScore.update({name : dicVal})
    #crosValScore.append(name + str(cross_val_score(val,X_train,Y_train,cv=kf).mean()))
    modelObjValues.append(val)
 
plt.figure(figsize=(8,6))
plt.title("Cross-Validation")
plt.bar(range(len(modelObjNames)), tempVal[0:len(tempVal)],color='maroon')
plt.xticks(range(len(modelObjNames)), modelObjNames[0:len(modelObjNames)], rotation='vertical')

print("Cross-Validation Degerleri : {}".format( crosValScore))

#önemli öznitelikler
#impFeatures = hepatitDataCorr['Category']
impFeatures = hepatitDataCorr[0]
featureNames = hepatitData.drop("Category", axis=1).columns
impFeatures=impFeatures[1:]
indices = np.argsort(impFeatures)[::-1]
impFeatures=impFeatures[indices]
featureNames = featureNames[indices]
impFeatures = [x for x in impFeatures]

plt.figure(figsize=(8,6))
plt.title("Önemli Öznitelikler")
plt.bar(range(len(featureNames)), impFeatures[0:len(impFeatures)],color=(0.5, 0.4, 0.7))
plt.xticks(range(len(featureNames)), featureNames[0:len(featureNames)], rotation='vertical')

#4e 3lük bir çizim alanı oluşturuyoruz.
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
for i, (name, model) in enumerate(modelsObjects.items()):

    row = i // 2
    col = i % 2
    disp = ConfusionMatrixDisplay.from_estimator(model, 
                                                 X_test, 
                                                 Y_test,
                                                 cmap='BuGn',
                                                 xticks_rotation='vertical',
                                                 display_labels=outputHepData,
                                                 ax=axs[row, col])
    disp.ax_.set_title(str(accuracies[i]))
plt.tight_layout()

print("%85 eğitim seti ile")
values=['Dogruluk(Accuracy)', 'Duyarlilik(Precision)','Hassasiyet(Recal)','F-measure','AUC']
df = pd.DataFrame(np.random.randn(7, 5),columns=values)
df.columns = [x for x in values]
df['Dogruluk(Accuracy)']=accuracies
df['Duyarlilik(Precision)']=precisionData
df['Hassasiyet(Recal)']=recallData
df['F-measure']=F1scoreData
df['AUC']=AUC
print(df.to_string())
print("\n--------------------------------------------------------")
print("%70 eğitim seti ile")
df70 = pd.DataFrame(np.random.randn(7, 5),columns=values)
df70.columns = [x for x in values]
df70['Dogruluk(Accuracy)']=accuracies70
df70['Duyarlilik(Precision)']=precisionData70
df70['Hassasiyet(Recal)']=recallData70
df70['F-measure']=F1scoreData70
df70['AUC']=AUC70
print(df70.to_string())

tempVal=[]
for i in (accuracies):
    tempVal.append(i[1])

loss_values=[]
for i in predList.values():
    loss_values.append(mean_absolute_error(Y_test,i))

pyplot.title('Accuracies')
pyplot.xlim(0, 1)
pyplot.ylim(0,1)
pyplot.plot(tempVal,scalex=False, label='Accuracies')
pyplot.legend()
pyplot.show()













