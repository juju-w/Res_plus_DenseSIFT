import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split,KFold
import pywt
import cyvlfeat as vlf
import joblib
from keras.datasets import cifar100,cifar10
import matplotlib.pyplot as plt
import os
import metrics
# def K_flod(data,k_fold,times):#path
#     # images=[os.path.join(path, imgs) for imgs in os.listdir(path)]
#     x = (len(data) // k_fold) * (k_fold - 1)
#     y=(len(data) // k_fold)
#     for i in range(times+1) :
#         if type(data)== np.ndarray:
#             temp=data[0:y]
#             data[0:y]=data[x:]
#             data[x:]=temp
#         else:
#             data.insert(0, data.pop())
#     return data[0:x],data[x:]

class data_get(object,):
    def __init__(self):
        self.image_size=48
        self.cifar_path="cifar-100-python"
        self.rootn='S:/study/celiac_disease_optical_image/data/videocapsule endoscopy images enlarge/normal'
        self.roots = 'S:/study/celiac_disease_optical_image/data/videocapsule endoscopy images enlarge/sick'
        # self.rootn='clear_data_image-wise/normal'
        # self.roots = 'clear_data_image-wise/sick'
        self.datan,self.datas=os.listdir(self.rootn),os.listdir(self.roots)
        self.data_path= np.array([os.path.join(self.rootn, self.datan[i]) for i in range(len(self.datan))]+[ os.path.join(self.roots, self.datas[i]) for i in range(len(self.datas))])
        self.lab=np.array([0 for index in range(len(self.datan))]+[1 for index in range(len(self.datas))])
    def provide_normal(self,j):
        data=np.empty((len(self.data_path),self.image_size,self.image_size,3),np.int)
        for i in range(len(self.data_path)):
            img=cv2.imread(self.data_path[i])
            data[i] = cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
            print(i)
        kf=KFold(n_splits=10)
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(data)
        np.random.seed(seed)
        np.random.shuffle(self.lab)
        list=[]
        for train, test in kf.split(data):
            list.append(train.tolist())
            list.append(test.tolist())
        X_train, X_test, y_train, y_test=data[list[2*j]],data[list[2*j+1]],self.lab[list[2*j]],self.lab[list[2*j+1]]
        return X_train, X_test,y_train,  y_test
    def provide_dsift(self):
        data=np.empty((len(self.data_path),self.image_size,self.image_size,3),np.int)
        dsift=np.empty((len(self.data_path),(self.image_size-9),(self.image_size-9),128),dtype=float)
        for i in range(len(self.data_path)):
            data[i]=img=cv2.imread(self.data_path[i])
            img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
            AH = np.concatenate([cA, cH], axis=1)
            VD = np.concatenate([cV, cD], axis=1)
            img = np.concatenate([AH, VD], axis=0)
            dsift[i] =vlf.sift.dsift(img)[1].reshape((self.image_size-9),(self.image_size-9),128)
        X_train, X_test, y_train, y_test = train_test_split(data, self.lab,random_state=1,test_size=0.3)
        Dsift_train, Dsift_test, _, _ = train_test_split(dsift, self.lab, random_state=1, test_size=0.3)
        return X_train, X_test, y_train, y_test,Dsift_train,Dsift_test
    def dsift_only(self,j):
        data=np.empty((len(self.data_path),(self.image_size-9)*(self.image_size-9)*128),dtype=float)
        for i in range(len(self.data_path)):
            img=cv2.imread(self.data_path[i],cv2.IMREAD_GRAYSCALE)
            # cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
            # AH = np.concatenate([cA, cH], axis=1)
            # VD = np.concatenate([cV, cD], axis=1)
            # img = np.concatenate([AH, VD], axis=0)
            data[i] =vlf.sift.dsift(img)[1].reshape(1,-1)
            print(i)
        kf=KFold(n_splits=10)
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(data)
        np.random.seed(seed)
        np.random.shuffle(self.lab)
        list=[]
        for train, test in kf.split(data):
            list.append(train.tolist())
            list.append(test.tolist())
        X_train, X_test, y_train, y_test=data[list[2*j]],data[list[2*j+1]],self.lab[list[2*j]],self.lab[list[2*j+1]]
        return X_train, X_test,y_train,  y_test

    def dsift_bow_kmeans(self,j,k):
        # import matplotlib.pyplot as plt
        self.k=k
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer()
        data=np.empty((len(self.data_path),self.image_size,self.image_size,3),np.int)
        for i in range(len(self.data_path)):
            img=cv2.imread(self.data_path[i])
            data[i]=cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        kf=KFold(n_splits=10)
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(data)
        np.random.seed(seed)
        np.random.shuffle(self.lab)
        list=[]
        for train, test in kf.split(data):
            list.append(train.tolist())
            list.append(test.tolist())
        X_train, X_test,train, test, y_train, y_test =data[list[2*j]],data[list[2*j+1]],self.data_path[list[2*j]],\
                                                      self.data_path[list[2*j+1]],self.lab[list[2*j]],self.lab[list[2*j+1]]
        dsift_train=np.empty((len(train),(self.image_size-9)*(self.image_size-9),128),dtype=float)
        fea_train=np.empty((len(train),self.k),dtype=float)
        dsift_test=np.empty((len(test),(self.image_size-9)*(self.image_size-9),128),dtype=float)
        fea_test=np.empty((len(test),self.k),dtype=float)
        #train
        for i in range(len(train)):
            img=cv2.imread(train[i],cv2.IMREAD_GRAYSCALE)
            # img=cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
            dsift_train[i] =vlf.sift.dsift(cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA))[1]
        kmean=MiniBatchKMeans(n_clusters=self.k,batch_size=256,init_size=self.k*3)
        bagtrain = dsift_train.reshape(len(train) * (self.image_size - 9) * (self.image_size - 9), 128)
        print('开始聚类')
        clubag=model_save_load('kms'+str(self.k)+"-"+str(j),kmean,bagtrain)
        print('train')
        for i in range(len(train)):
            print(i)
            list1 = clubag.predict(bagtrain[i * (self.image_size - 9) * (self.image_size - 9):(i + 1) * (self.image_size - 9) * (self.image_size - 9)])
            for j in list1:
                fea_train[i][j] +=1
        fea_train =train_data =transformer.fit_transform(fea_train).toarray()
        for i in range(len(test)):
            img=cv2.imread(test[i],cv2.IMREAD_GRAYSCALE)
            # img=cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
            dsift_test[i] =vlf.sift.dsift(cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA))[1]
        bagtest = dsift_test.reshape(len(test) * (self.image_size - 9) * (self.image_size - 9), 128)
        print('test' )
        for i in range(len(test)):
            print(i)
            list1 = clubag.predict(bagtest[i * (self.image_size - 9) * (self.image_size - 9):(i + 1) * (self.image_size - 9) * (self.image_size - 9)])
            for j in list1:
                fea_test[i][j] +=1
        # fea_test = test_data= transformer.fit_transform(fea_test).toarray()
        # train_labels=y_train
        # test_labels=y_test
        # import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # from sklearn import svm
        # from fast_tsne import fast_tsne as ft
        # svm = svm.SVC()
        # svm = model_save_loadxy(str(k) + str(j), svm, train_data.reshape([train_data.shape[0], -1]), train_labels)
        # p = svm.predict(test_data.reshape([test_data.shape[0], -1])).reshape(-1, 1).tolist()
        # t = test_labels.reshape(-1, 1).tolist()
        # print(str(metrics.accuracy(t, p)))
        # # with open('dsiftbowtest.txt', 'a') as f:  # 设置文件对象
        # #     f.write(str(k) + '-' + str(j) + ',' + str(metrics.accuracy(t, p)) + ',' + str(
        # #         metrics.precision(t, p)) + ',' + str(metrics.recall(t, p)) + ',' + str(metrics.f1score(t, p))
        # #             + ',' + str(metrics.ft(t, p)) + '\n')
        # x2dtmp=PCA(n_components=50).fit_transform(fea_train)
        # x2d = ft(x2dtmp)
        # b = np.where(np.array(y_train) == 1)
        # a = np.where(np.array(y_train) == 0)
        # plt.scatter(x2d[a, 0], x2d[a, 1], c='r')
        # plt.scatter(x2d[b, 0], x2d[b, 1], c='b')
        # plt.show()
        return  X_train,X_test,y_train, y_test,fea_train,fea_test



def model_save_load(name, model, x):
    model_name = name + '.pkl'
    if model_name not in os.listdir('data/model'):
        model.fit(x)
        joblib.dump(model, 'data/model/' + model_name)
        return model
    else:
        model = joblib.load('data/model/' + model_name)
        return model

def model_save_loadxy(name, model, x,y=None):
    model_name = name + '.pkl'
    if model_name not in os.listdir('data/model'):
        model.fit(X=x,y=y)
        joblib.dump(model, 'data/model/' + model_name)
        return model
    else:
        model = joblib.load('data/model/' + model_name)
        return model
