import cnn
import data
import plot

if __name__ == '__main__':
    train_generator, valid_generator, test_generator = data.get_image_generators()

    num_classes = 102  # Counting BACKGROUND Class
    input_shape = (300, 200, 3)

    epochs = 20

    model = cnn.get_cnn_sequential_model(num_classes, input_shape)
    cnn.compile_cnn(model)

    history = cnn.fit_generator(model, train_generator, valid_generator, epochs)
    plot.draw_history(history)

    score = cnn.evaluate_generator(test_generator, test_generator)

    print('\nTest loss: ' + str(score[0]))
    print('Test accuracy: ' + str(score[1]))
