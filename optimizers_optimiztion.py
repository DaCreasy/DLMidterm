import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
# import matplotlib.pyplot as plt
import numpy as np
import sys

from TimeHistory import TimeHistory

# Hyperparameters for training
batch_size = 32
num_classes = 100
epochs = 1
print("Chosen batch size: %s" % batch_size)
print("Chosen number of epochs: %s" % epochs)

# Hyperparameters for the optimizer
opts = ['rms', 'sgd', 'adagrad', 'adadelta', 'adam']
numOpts = len(opts)
opts_acc = [None]*numOpts
opts_loss = [None]*numOpts
opts_time = [None]*numOpts

numTrials = 5
agr_opts_loss = [None]*numTrials
agr_opts_acc = [None]*numTrials
agr_opts_loss_name = [None]*numTrials
agr_opts_acc_name = [None]*numTrials
agr_opts_time = [None]*numTrials
agr_opts_time_name = [None]*numTrials
time_callback = TimeHistory()

for j in range(numTrials):
    for i in range(numOpts):
        

        opt = opts[i] # choose from list above, more can be added
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


        (x_train,y_train),(x_test,y_test)=cifar100.load_data()
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
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

        model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True, callbacks=[time_callback])
        times = time_callback.times
        print(times)

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        # history_dict = history.history

        # for making some graphs for the paper
        # plt.plot(range(epochs), history_dict['loss'], label='Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Performance')
        # plt.legend()

        opts_loss[i] = scores[0]
        opts_acc[i] = scores[1]
        opts_time[i] = np.average(times)

    print('Best loss for trial ', j, ': ', opts[opts_loss.index(min(opts_loss))], ' with ', min(opts_loss))
    print('Best accuracy for trial ', j, ': ', opts[opts_acc.index(max(opts_acc))], ' with ', max(opts_acc))
    print('Best time for trial ', j, ': ', opts[opts_time.index(min(opts_time))], ' with ', min(opts_time))

    agr_opts_loss[j] = min(opts_loss)
    agr_opts_loss_name[j] = opts[opts_loss.index(agr_opts_loss[j])]
    agr_opts_acc[j] = max(opts_acc)
    agr_opts_acc_name[j] = opts[opts_acc.index(agr_opts_acc[j])]
    agr_opts_time[j] = min(opts_time)
    agr_opts_time_name[j] = opts[opts_time.index(agr_opts_time[j])]

print(agr_opts_loss_name)
print(agr_opts_loss)
print()
print(agr_opts_acc_name)
print(agr_opts_acc)
print()
print(agr_opts_time_name)
print(agr_opts_time)
    
sys.stdout =  open('output.txt', 'w')
print(agr_opts_loss_name)
print(agr_opts_loss)
print(agr_opts_acc_name)
print(agr_opts_acc)
print(agr_opts_time_name)
print(agr_opts_time)
