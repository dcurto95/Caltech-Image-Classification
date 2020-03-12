import keras
from keras import initializers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Sequential, load_model


def get_cnn_sequential_model(num_classes, input_shape):
    model = Sequential()

    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        input_shape=input_shape,
        activation='relu',
        kernel_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.055, seed=34),
        bias_initializer=initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=52)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(100, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(50, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(num_classes, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    print(model.summary())
    return model


def compile_cnn(model, optimizer='adam'):
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy'])


def compile_cnn_parametrized(model, loss, optimizer):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy'])


def fit_cnn(model, x_train, y_train, x_validation, y_validation, batch_size, epochs,
            class_weights=[]):  # , tbCallBack):
    if not class_weights:
        return model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_validation, y_validation))
    return model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_validation, y_validation),
        class_weight=class_weights)


def fit_generator(model, train_generator, valid_generator, epochs):
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    return model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        validation_data=valid_generator)


def transform_y_to_categorical(y, num_classes):
    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)

    return y


def save_model_to_disk(model, current_class, area):
    model.save("model_" + str(area) + "_" + str(current_class) + ".h5")
    """# serialize model to JSON
    model_json = model.to_json()
    with open("model_" + str(area) + "_" + str(current_class) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")"""
    print("Saved model to disk")


def load_model_from_disk(current_class, area):
    loaded_model = load_model("model_" + str(area) + "_" + str(current_class) + ".h5")
    """# load json and create model
    json_file = open("model_" + str(area) + "_" + str(current_class) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")"""
    return loaded_model


def predict(model, x_evaluation, batch_size):
    return model.predict(x_evaluation, batch_size=batch_size)


def predict_classes(model, x_evaluation, batch_size):
    return model.predict_classes(x_evaluation, batch_size=batch_size)


def evaluate_generator(model, test_generator):
    return model.evaluate_generator(test_generator, steps=test_generator.samples // test_generator.batch_size,
                                    verbose=0)
