from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

#read data
train_data=np.loadtxt('training_data.csv',skiprows=1,delimiter=',')
test_data=np.loadtxt('evaluation_data.csv',skiprows=1,delimiter=',')

label=train_data[:,-1]
train_data=train_data[:,:-1]
#network
model = Sequential([
    Dense(6, input_shape=(6,)),
    Activation('relu'),
    Dense(24),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#trainning
model.fit(train_data, label.astype(int), epochs=50, batch_size=4)

#test
test_predict=model.predict(test_data, batch_size=None, verbose=0, steps=None)

label_predict=test_predict>0.9
label_predict= label_predict.astype(int)
np.savetxt('labels.txt', label_predict, delimiter=',')
np.savetxt('labels.csv', label_predict, delimiter=',')