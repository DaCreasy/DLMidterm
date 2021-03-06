import keras
from keras.datasets import cifar100, cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, Conv2DTranspose, SeparableConv2D
from keras.layers import UpSampling2D, Cropping2D, AveragePooling2D
import sys

# Hyperparameters for training
batch_size = 16
num_classes = 10
epochs = 25
print("Chosen batch size: %s" % batch_size)
print("Chosen number of epochs: %s" % epochs)

# Hyperparameters for the optimizer
opts = ['rms', 'sgd', 'adagrad', 'adadelta', 'adam']
opt = opts[4] #  choose from list above, more can be added
print("Chosen optimizer: %s with..." % opt)
learning_rate = 0.0001
decay = 1e-6
print("\t learning rate of %s and..." % learning_rate)
print("\t decay rate of %s" % decay)
if opt=='rms':
    optimizer = keras.optimizers.rmsprop(lr=learning_rate, decay=decay)  
elif opt=='sgd':
    optimizer = keras.optimizers.sgd(lr=learning_rate, decay=decay)
elif opt=='adagrad':
    optimizer = keras.optimizers.adagrad(lr=learning_rate, decay=decay)
elif opt=='adadelta':
    optimizer = keras.optimizers.adadelta(lr=learning_rate, decay=decay)
elif opt=='adam':
    optimizer = keras.optimizers.adam(lr=learning_rate, decay=decay)
    

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print(x_train.shape)
# (50000, 32, 32, 3), 50000 images, 32x32pixels, 3 color channels(RGB)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Normalize input values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()

# model.add(Cropping2D(cropping=((2,2),(4,4)), input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2DTranspose(32, (3,3), padding='same'))
# model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(UpSampling2D(size=(2,2)))
model.add(Activation('elu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2DTranspose(64, (3,3), padding='same'))
# model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# model.add(Cropping2D(cropping=((2,2),(2,2))))
# model.add(SeparableConv2D(64, (3,3), strides=(1,1), padding='valid'))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=1)
sys.stdout = open('output2.txt', 'w')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
