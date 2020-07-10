#!/usr/bin/env python3
import argparse
import json
import math
import os
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans
from PostQuantizer import ASYMMETRIC

from model_definition import image_size, preprocess_imagenet
import utilities as util
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_model(num_classes):
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = MobileNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    x = Dense(num_classes, activation='softmax')(x)  # final layer with softmax activation

    updatedModel = Model(base_model.input, x)

    return updatedModel


def compile_model(compiledModel):
    compiledModel.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=Adadelta(),
                          metrics=['accuracy'])

def snap(tensor,num_bits):
    shape = tensor.shape
    
    u = tensor.reshape(-1,1)
    kmeans = KMeans(n_clusters=(2**num_bits))
    kmeans.fit(u)
    
    grid = kmeans.cluster_centers_
    grid = sorted(grid)
    
    break_points = [0]*(len(grid)-1)
    
    for i in range(len(grid)-1):
        break_points[i] = (grid[i] + grid[i+1]) / 2


    temp = np.ndarray.flatten(tensor)
    temp = list(temp)
    right_end = break_points[-1]

    for idx_t, pt_t in enumerate(temp):
        if pt_t > right_end:
            temp[idx_t] = grid[-1]
        else:
            for idx_b, pt_b in enumerate(break_points):
                if pt_t < pt_b:
                    temp[idx_t] = grid[idx_b]
                    break

    temp = np.array(temp)
    temp = temp.reshape(shape)
    return temp


def modelFitGenerator():

    train_datagen = ImageDataGenerator(
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.4,
        preprocessing_function=preprocess_imagenet
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_imagenet
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True,
        interpolation='lanczos'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', shuffle=True,
        interpolation='lanczos'
    )

    num_train_samples = len(train_generator.classes)
    num_valid_samples = len(validation_generator.classes)

    num_train_steps = math.floor(num_train_samples / batch_size)
    num_valid_steps = math.floor(num_valid_samples / batch_size)

    train_classes = len(set(train_generator.classes))
    test_classes = len(set(validation_generator.classes))

    if train_classes != test_classes:
        print('number of classes in train and test do not match, train {}, test {}'.format(train_classes, test_classes))
        exit(1)

    # save class names list before training

    label_map = train_generator.class_indices
    class_idx_to_label = {v: k for k, v in label_map.items()}
    labels = []
    for i in range(len(class_idx_to_label)):
        label = class_idx_to_label[i]
        labels.append(label)

    labels_txt = u"\n".join(labels)
    with open(output_class_names_path, 'w') as classes_f:
        classes_f.write(labels_txt)
    print("Saved class names list file to {}".format(output_class_names_path))

    fitModel = get_model(num_classes=train_classes)

    # fitModel.load_weights('trained_model/baseline/model.h5')
    fitModel.load_weights('trained_model/model.h5')
    # fitModel.load_weights('.mdl_wts.h5')

    # quantizer = ASYMMETRIC()

    # for layer in fitModel.layers:
    #     t = layer.get_weights()
    #     for idx, tensor in enumerate(t):
    #         if len(tensor.shape)==4:
    #             print(layer.name, tensor.shape)
    #             # if 'depthwise' in layer.name:
    #             if True:
    #                 new_tensor = snap(tensor,args.bits)
    #                 # new_tensor = quantizer.quantize(new_tensor,bits=args.bits)['dequant_array']
    #             else:
    #                 new_tensor = quantizer.quantize(tensor,bits=args.bits)['dequant_array']
    #             t[idx] = new_tensor
    #         if len(tensor.shape)==2:
    #             print('this tensor has shape 2:',layer.name, tensor.shape)
    #             # new_tensor = snap(tensor,args.bits)
    #             new_tensor = quantizer.quantize(tensor,bits=args.bits)['dequant_array']
    #             t[idx] = new_tensor
    #     layer.set_weights(t)



    compile_model(fitModel)
    fitModel.evaluate(validation_generator)

    # fitModel.save('selective_kmean.h5')
    # fitModel.save('kmean.h5')


def main():
    modelFitGenerator()


if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='workspace/datasets/open-images-10-classes/train/',
        required=False,
        help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
    )
    parser.add_argument(
        '--eval_dataset_path',
        type=str,
        default='workspace/datasets/open-images-10-classes/eval/',
        required=False,
        help='Path to folders of labeled eval images'
    )
    parser.add_argument(
        '--output_model_path',
        type=str,
        default='trained_model',
        required=False,
        help='Where to save the trained model.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs, full passes through the dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size. Number of images to process at each gradient descent step.'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=8,
        help='lambda value'
    )

    args = parser.parse_args()
    train_data_dir = args.dataset_path
    validation_data_dir = args.eval_dataset_path
    nb_epoch = args.epochs
    batch_size = args.batch_size
    output_model_path = args.output_model_path
    output_class_names_path = os.path.join(output_model_path, 'class_names.txt')

    os.makedirs(output_model_path, exist_ok=True)

    with open(os.path.join(output_model_path,'model_schema.json'), 'w') as schema_f:
        schema_f.write(json.dumps({
            "output_names": "dense_3/Softmax",
            "input_names": "input_1",
            "preprocessor": "imagenet",
            "input_shapes": "1,224,224,3",
            "task": "classifier",
            "dataset": "custom"
        }, indent=4))

    output_model_path = os.path.join(output_model_path, 'model.h5')

    main()
