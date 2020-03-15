import os

import PIL
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def create_data_splits():
    # data = {}
    data_images = []
    data_classes = []

    num_classes = 0

    for object_class in os.listdir("../dataset/101_ObjectCategories/"):
        # data[object_class] = []
        num_classes += 1
        image_class_path = "../dataset/101_ObjectCategories/" + object_class + "/"

        for image in os.listdir(image_class_path):
            im = Image.open(image_class_path + image)
            width, height = im.size
            if im.mode != "RGB":
                im = im.convert('RGB')
            # TODO: Check with and without rotation
            if height > width:
                im = im.rotate(90)
            im = im.resize((300, 200), resample=PIL.Image.LANCZOS)
            # data[object_class].append(np.asarray(im))
            data_classes.append(object_class)
            data_images.append(im)
    input_shape = np.asarray(im).shape

    x_train, x_test, y_train, y_test = train_test_split(data_images, np.asarray(data_classes),
                                                        test_size=0.4, random_state=32,
                                                        stratify=np.asarray(data_classes))
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=32,
                                                                  stratify=y_test)

    for i, (image, im_class) in enumerate(zip(x_train, y_train)):
        directory = "../dataset/Train/" + im_class + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(directory + str(i) + ".jpg")

    for i, (image, im_class) in enumerate(zip(x_validation, y_validation)):
        directory = "../dataset/Valid/" + im_class + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(directory + str(i) + ".jpg")

    for i, (image, im_class) in enumerate(zip(x_test, y_test)):
        directory = "../dataset/Test/" + im_class + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        image.save(directory + str(i) + ".jpg")


"""
def load_data():
    # data = {}
    data_images = []
    data_classes = []

    num_classes = 0

    for object_class in os.listdir("../dataset"):
        # data[object_class] = []
        num_classes += 1
        image_class_path = "../dataset/" + object_class + "/"

        for image in os.listdir(image_class_path):
            im = Image.open(image_class_path + image)
            width, height = im.size
            if im.mode != "RGB":
                im = im.convert('RGB')
            # TODO: Check with and without rotation
            if height > width:
                im = im.rotate(90)
            im = im.resize((300, 200), resample=PIL.Image.LANCZOS)
            # data[object_class].append(np.asarray(im))
            data_classes.append(object_class)
            data_images.append(np.asarray(im))
    input_shape = np.asarray(im).shape
    
    return num_classes, input_shape, np.asarray(data_classes), np.asarray(data_images)
"""


# 9145 images
def get_image_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    valid_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    train_generator = train_datagen.flow_from_directory(
        "../dataset/Train/",
        target_size=(300, 200),
        batch_size=128,
        # color_mode="rgb",
        class_mode='categorical')
    valid_generator = valid_datagen.flow_from_directory(
        "../dataset/Valid/",
        target_size=(300, 200),
        batch_size=128,
        # color_mode="rgb",
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        "../dataset/Test/",
        target_size=(300, 200),
        batch_size=128,
        # color_mode="rgb",
        class_mode='categorical')

    return train_generator, valid_generator, test_generator


if __name__ == '__main__':
    create_data_splits()
