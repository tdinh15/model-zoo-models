#!/usr/bin/env python3
import argparse
import json
import math
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
                          optimizer=SGD(learning_rate=0.01,momentum=0.9,nesterov=True),
                          metrics=['accuracy'])

class QuantizeWeights(keras.callbacks.Callback):
    def __init__(self):
        self.quantizer = ASYMMETRIC()

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            t = layer.get_weights()
            for idx, tensor in enumerate(t):
                if len(tensor.shape) in [2,4]:
                    new_tensor = self.quantizer.quantize(tensor,bits=args.bits)['dequant_array']
                    t[idx] = new_tensor
            layer.set_weights(t)

class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        # if epoch in LR_SCHEDULE:
        print("Epoch: ", epoch, " Learning rate:", scheduled_lr)

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (40, 0.1),
    (70, 0.01),
    (100, 0.001),
    (130, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0]:
        return lr
    for i in range(1,len(LR_SCHEDULE)):
        if epoch < LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i-1][1]
    return LR_SCHEDULE[-1][1]

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
    
    fitModel.save('model.h5')
    quantizer = {
            "class_name": "QARegularizer",
            "config": {
                "num_bits": 4,
                "lambda_1": 0.0,
                "lambda_2": float(args.lamb2),
                "lambda_3": float(args.lamb3),
                "lambda_4": 0.0,
                "lambda_5": 0.0,
                "quantizer_name": "asymmetric"
                }
            }
    layer_list = util.list_tf_keras_model('model.h5')
    for layer_name, layer_attr in layer_list.items():
        if 'kernel_regularizer' in layer_attr:
            layer_attr['kernel_regularizer'] = quantizer
        if 'depthwise_regularizer' in layer_attr:
            layer_attr['depthwise_regularizer'] = quantizer
    
    fitModel = util.attach_regularizers(
            os.path.join("model.h5"), 
            layer_list,
            target_keras_h5_file=None, 
            verbose=False, 
            backend_session_reset=True,)

    fitModel.load_weights('keras.h5')

    # for layer in fitModel.layers:
    #     if 'depthwise' in layer.name:
    #         weight_list = layer.get_weights()
    #         if len(weight_list) == 1:
    #             tensor = weight_list[0]
    #             new_tensor = np.clip(tensor,-0.75,0.75)
    #             layer.set_weights([new_tensor])

    # for layer in fitModel.layers:
    #     # weights = layer.get_weights()
    #     # for wt in weights:
    #     #     print(wt.max() - wt.min())
    #     if 'depthwise' in layer.name:
    #         weight_list = layer.get_weights()
    #         if len(weight_list) == 1:
    #             tensor = weight_list[0]
    #             print(layer.name)
    #             print(tensor.max() - tensor.min())

    compile_model(fitModel)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
    fitModel.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=num_valid_steps,
        callbacks=[mcp_save,
                # CustomLearningRateScheduler(lr_schedule),
                QuantizeWeights(),
                reduce_lr_loss,
                # earlyStopping,
                ]
    )

    fitModel.save(output_model_path, include_optimizer=False)
    print("Saved trained model to {}".format(output_model_path))


def main():
    modelFitGenerator()


if __name__ == '__main__':
    # constants

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        required=True,
        help='Path to folders of labeled images. Expects "train" and "eval" subfolders'
    )
    parser.add_argument(
        '--eval_dataset_path',
        type=str,
        default=None,
        required=True,
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
        default=1,
        help='Number of training epochs, full passes through the dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size. Number of images to process at each gradient descent step.'
    )
    parser.add_argument(
        '--lamb2',
        type=float,
        default=0,
        help='lambda value'
    )
    parser.add_argument(
        '--lamb3',
        type=float,
        default=0,
        help='lambda value'
    )
    parser.add_argument(
        '--bits',
        type=float,
        default=4,
        help='number of bits to quantize'
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
