# Import des libraries classique (numpy, pandas, ...)
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

df=pd.read_csv("data.csv")
df=df.stack().str.replace(',','.').unstack()
print("number of lines",df.shape[0])
print("number of empty or Nan value ", df.isnull().sum().sum())

df_nan=df.dropna(subset=['Experience'])
df_nan['Experience']=df_nan['Experience'].astype('float')
df_datascientist=df_nan[df_nan['Metier'].str.contains("Data scientist")==True ]
df_dataengineer=df_nan[df_nan['Metier'].str.contains("Data engineer")==True ]

m_datascientist=np.median(df_datascientist['Experience'])
m_dataengineer=np.mean(df_dataengineer['Experience'])
print('median exp datascientist',m_datascientist)
print('meean exp dataengineer',m_dataengineer)

df['Experience'].loc[(df['Metier'] =='Data scientist') & (df['Experience'].isnull() == True )]=m_datascientist
df['Experience'].loc[(df['Metier'] =='Data engineer') & (df['Experience'].isnull() == True )]=m_dataengineer
df['Experience']=df['Experience'].astype('float')
print(np.mean(df['Experience']))
a_me=df.groupby(['Metier']).mean()
print('mean for each metier')
print(a_me)
a_me.plot(kind='bar',rot=0)
plt.title("Le nombre moyen d'années d'expériences pour chaque métier")
#plt.show()
a=df.groupby(['Metier'])

df['Exp_label'] = ""
for job, data  in a:
    mean=np.mean(data['Experience'])
    std=np.std(data['Experience'])/2

    print(mean)
    print(std)
    print(str(job))
    df['Exp_label'].loc[(df['Metier'] ==str(job)) & (df['Experience']>mean+std)]="expert"
    df['Exp_label'].loc[(df['Metier'] == job) & (df['Experience'] < mean + std)] = "avancé"
    df['Exp_label'].loc[(df['Metier'] == job) & (df['Experience'] < mean )] = "confirmé"
    df['Exp_label'].loc[(df['Metier'] == job) & (df['Experience'] < mean - std)] = "debutant"

#df=df.stack().str.replace('/',' ').unstack()
all=df['Technologies'].str.split('/').sum()

tech=pd.DataFrame(all, columns=['tech'])
tech['num']=1
t=tech.groupby(['tech']).count()


print(t)
t=t.sort_values('num', ascending=False)
print(t)
t[0:5].plot(kind='bar',rot=0)
plt.title("Les 5 technologies les plus utilisées")
#plt.show()

#cluster sur technologie
df_nan=df.dropna(subset=['Metier'])

vectorizer = TfidfVectorizer(stop_words='english')
#print(df_encoded)
Xdf_nan=df_nan['Diplome']+df_nan['Exp_label']+df_nan['Technologies']
#X=vectorizer.fit_transform(fruit_data)
X_train = vectorizer.fit_transform(Xdf_nan)

Y_train = vectorizer.fit_transform(df_nan['Metier'])

le = LabelEncoder()
le.fit(df_nan['Metier'])
Y_train=le.transform(df_nan['Metier'])
Xdf=df['Diplome']+df['Exp_label']+df['Technologies']

#X_train=le2.transform(Xdf_nan)
#X=le2.transform(Xdf)
X_train=vectorizer.fit_transform(Xdf_nan)
#4 clusters
true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X_train,Y_train)

X=vectorizer.fit_transform(Xdf)
#creation d'une colone metier predit
df['Predict_Metier'] = model.predict(X[:,0:np.size(X_train,1)])
Y_test=model.predict(X_train)
accurate=accuracy_score(Y_train, Y_test)*100
print('result %.2f'%accurate)

