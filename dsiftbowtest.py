from sklearn.svm import SVC
from data_provider import data_get as dg
import joblib
import os
import metrics

k=2500
dg=dg()
j=1
dg.dsift_bow_kmeans(j,k)