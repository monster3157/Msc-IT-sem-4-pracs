from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
dataset=loadtxt('pima-indians-diabetes.csv', delimiter=',')
dataset
X=dataset[:,0:8]
Y=dataset[:,8]
X
Y

model=Sequential()
model.add (Dense (12, input_dim=8, activation= 'relu'))
model.add(Dense (8,activation='relu'))
model.add(Dense (1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,Y, epochs=150, batch_size=10)
_,accuracy=model.evaluate(X,Y)
print('Accuracy of model is', (accuracy*100))

prediction=model.predict_classes(X)
exec("for i in range(5): print(X[i].tolist(), prediction[i],Y[i])")

