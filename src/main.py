import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import cnn
import data

if __name__ == '__main__':
    train_generator, valid_generator, test_generator = data.get_image_generators()

    num_classes = 102  # Counting BACKGROUND Class
    input_shape = (300, 200, 3)

    epochs = 20
    batch_size = 10000

    model = cnn.get_cnn_sequential_model(num_classes, input_shape)
    cnn.compile_cnn(model)

    cnn.fit_generator(model, train_generator, valid_generator, batch_size, epochs)

    score = model.evaluate_generator(test_generator, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
