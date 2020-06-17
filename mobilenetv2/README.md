# Getting Started with MobilenetV2

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd mobilenetv2```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:


# Download pretrained model on Open Images 10 Classes
```bash
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes
```
# Download pretrained imagenet model
```bash
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet
```
# Download dataset for Transfer Learning training
```bash
./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval
```
# Train

(Set --epochs and --batch_size to 1 for a quick training run.)
```bash
./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 150
```
# Evaluate a trained model
```bash
./dev_docker_run ./eval.py --dataset_path workspace/datasets/open-images-10-classes/eval/ --input_model_path trained_model/model.h5
```
# Demo

This runs inference on a single image.
```bash
./dev_docker_run ./demo.py --input_model_path trained_model/model.h5 --image_file test_images/dog.jpg
```
# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

Open Image 10 Classes Commands

### Preparation
```bash
leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv2-oi
mkdir mobilenetv2-oi
mkdir mobilenetv2-oi/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path mobilenetv2-oi/baselineFp32Results --framework tf --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --quantizer asymmetric --bits 8 --output_path mobilenetv2-oi/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir mobilenetv2-oi/compiled_lre_fp32
leip compile --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --output_path mobilenetv2-oi/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-oi/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path mobilenetv2-oi/compiled_lre_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir mobilenetv2-oi/compiled_lre_int8
leip compile --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --output_path mobilenetv2-oi/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path mobilenetv2-oi/compiled_lre_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
```
### Convert model to integer
```bash
echo workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg > rep_dataset.txt
mkdir mobilenetv2-oi/tfliteOutput
leip convert --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --framework tflite --output_path mobilenetv2-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset rep_dataset.txt
```
### LRE Int8 (full)
```bash
leip compile --input_path mobilenetv2-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt --preprocessor ''
```

## Imagenet Commands

To download ImageNet and prepare images for evaluation, see instructions here: https://github.com/latentai/model-zoo-models#user-content-imagenet-download--eval-preparation

### Preparation
```bash
leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet
rm -rf mobilenetv2-imagenet
mkdir mobilenetv2-imagenet
mkdir mobilenetv2-imagenet/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path mobilenetv2-imagenet/baselineFp32Results --framework tf --input_path workspace/models/mobilenetv2/keras-imagenet --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/mobilenetv2/keras-imagenet --quantizer asymmetric --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir mobilenetv2-imagenet/compiled_lre_fp32
leip compile --input_path workspace/models/mobilenetv2/keras-imagenet --output_path mobilenetv2-imagenet/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-imagenet/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path mobilenetv2-imagenet/compiled_lre_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir mobilenetv2-imagenet/compiled_lre_int8
leip compile --input_path workspace/models/mobilenetv2/keras-imagenet --output_path mobilenetv2-imagenet/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/compiled_lre_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt
```
### Convert model to integer
```bash
echo /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG > rep_dataset.txt
mkdir mobilenetv2-imagenet/tfliteOutput
leip convert --input_path workspace/models/mobilenetv2/keras-imagenet --framework tflite --output_path mobilenetv2-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset rep_dataset.txt
```
### LRE Int8 (full)
```bash
leip compile --input_path mobilenetv2-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --preprocessor ''
```