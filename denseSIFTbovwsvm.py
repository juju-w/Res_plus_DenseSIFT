from sklearn.svm import SVC
from data_provider import data_get as dg
import joblib
import os
import metrics

def model_save_load(name, model, x,y=None):
    model_name = name + '.pkl'
    if model_name not in os.listdir('data/model'):
        model.fit(X=x,y=y)
        joblib.dump(model, 'data/model/' + model_name)
        return model
    else:
        model = joblib.load('data/model/' + model_name)
        return model
dg=dg()
for i in range(10):
    for j in range(10):
        train_data, test_data, train_labels, test_labels = dg.dsift_only(j)
        svm=SVC(kernel='poly',degree=3)
        svm=model_save_load('svm'+str(i)+'-'+str(j),svm,train_data.reshape([train_data.shape[0],-1]),train_labels)
        plab=p=svm.predict(test_data.reshape([test_data.shape[0],-1])).reshape(-1,1).tolist()
        t=test_labels.reshape(-1,1).tolist()
        print(str(metrics.accuracy(t, p)))
        with open('denseSIFTsvm.txt', 'a') as f:  # 设置文件对象
            f.write(str(i) + '-' + str(j) + ',' + str(metrics.accuracy(t, p)) + ',' + str(
                metrics.precision(t, p)) + ',' + str(metrics.recall(t, p)) + ',' + str(metrics.f1score(t, p))
                    + ',' + str(metrics.ft(t, p)) + '\n')


