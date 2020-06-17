# Getting Started with LeNet GTC

This model is trained with LEIP SDK, therefore the SDK should be installed to train this model.

Start by cloning this repo:
* ```git clone https://github.com/latentai/model-zoo-models.git```
* ```cd lenet_gtc```

Once this repo is cloned locally, you can use the following commands to explore LEIP framework:

# MNIST dataset

This code uses the built-in data library MNIST for training.
It is downloaded automatically at the first run, and is used afterwards.

For evaluation, please download this dataset:

```
leip zoo download --dataset_id mnist --variant_id eval 
```

# Train a new model from scratch

Training of the model is performed "from scratch". Since the model is simple and the dataset is relatively small, the whole training might take around 30 seconds for one epoch.
10 epochs is usually enough to get a decent result.

Basic training commands:

`python3 train.py --basedirectory train_dir`
which is equivalent to the following default parameters:
`python3 train.py --basedirectory train_dir --batch_size 64 --learning_rate 0.0002 --lambda_bit_loss 1e-5 --lambda_distillation_loss 0.01 --lambda_regularization 0.01`

ADAM optimization is used.

Important moment to take into account is `lambda_bit_loss` and `lambda_distillation_loss`. These two arguments control the quantization process. Their values are chosen so that a model accurate enough and with low number of bits is created! 
Increasing `lambda_distillation_loss` loss increases accuracy.
Increasing `lambda_bit_loss` decreases average number of bits used.
The regularization term, determined by `lambda_distillation_loss` pulls all the weights down towards zero. Very often it helps the GTC model converge to a better local solution. It is not meaningful for the LeNet system and is presented here as a demonstration of system capabilities.

If the values of the lambdas are too high, the system can not attain high accuracy, so it is not recommended to increase them by more then 20 times.

Expected result after 5 epochs:
```('lp_accuracy', 'dense_2') 0.94
('hp_accuracy', 'dense_2') 0.97
('total_loss', 'total_loss') 0.11
bit_loss 130.15
('distillation_loss', 'dense_2') 0.20
('hp_cross_entropy', 'dense_2') 0.08
regularization_term 202.27
```
The results are located in `./train_dir/lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01`
which will be created during the run. Main directories there:
* `training_model_final` - contains the final trained HP model
* `int_model_final`      - contains the final trained LP model

These two models can be further taken through the leip pipeline and the evaluations and the compilations scripts, explained next.

For the next steps, run `cd train_dir/lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01`

# Evaluate a trained model
We can use python script running inferences from a loaded model on images. The following command
```
python3 eval.py --input_path ./pretrained_models/fp32_model/
```
will load HP model and run it on mnist test set, producing a list of results. You can replace the directory `fp32_model` with `int_model`.


# Demo

This runs inference on a single image.

```
python3 demo.py --image_file mnist_examples/images/eight/eight.jpg  --input_model pretrained_models/int_model/
```
It will load pretrained model and apply to a 28x28 image.


# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

## High-Precision GTC Model

### Preparation
```bash
leip zoo download --model_id lenet_gtc --variant_id high_precision
leip zoo download --dataset_id mnist --variant_id eval
rm -rf lenet_hp_evaluate_variants
mkdir lenet_hp_evaluate_variants
mkdir lenet_hp_evaluate_variants/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path lenet_hp_evaluate_variants/baselineFp32Results --framework tf --input_path workspace/models/lenet_gtc/high_precision --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/lenet_gtc/high_precision --quantizer asymmetric --bits 8 --output_path lenet_hp_evaluate_variants/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir lenet_hp_evaluate_variants/compiled_lre_fp32
leip compile --input_path workspace/models/lenet_gtc/high_precision --output_path lenet_hp_evaluate_variants/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path lenet_hp_evaluate_variants/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path lenet_hp_evaluate_variants/compiled_lre_fp32/bin --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir lenet_hp_evaluate_variants/compiled_lre_int8
leip compile --input_path workspace/models/lenet_gtc/high_precision --output_path lenet_hp_evaluate_variants/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path lenet_hp_evaluate_variants/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path lenet_hp_evaluate_variants/compiled_lre_int8/bin --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### Convert model to integer
```bash
echo workspace/datasets/mnist/eval/images/seven/5576.jpg > rep_dataset.txt
mkdir lenet_hp_evaluate_variants/tfliteOutput
leip convert --input_path workspace/models/lenet_gtc/high_precision --framework tflite --output_path lenet_hp_evaluate_variants/tfliteOutput --data_type int8 --policy TfLite --rep_dataset rep_dataset.txt
```
### LRE Int8 (full)
```bash
leip compile --input_path lenet_hp_evaluate_variants/tfliteOutput/model_save/inference_model.cast.tflite --output_path lenet_hp_evaluate_variants/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path lenet_hp_evaluate_variants/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path lenet_hp_evaluate_variants/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt --preprocessor rgbtogray_int8
```
## Low-Precision GTC Model

### Preparation
```bash
leip zoo download --model_id lenet_gtc --variant_id low_precision
leip zoo download --dataset_id mnist --variant_id eval
rm -rf lenet_lp_evaluate_variants
mkdir lenet_lp_evaluate_variants
mkdir lenet_lp_evaluate_variants/baselineFp32Results
```
### Original FP32
```bash
leip evaluate --output_path lenet_lp_evaluate_variants/baselineFp32Results --framework tf --input_path workspace/models/lenet_gtc/low_precision --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### LEIP Compress ASYMMETRIC
```bash
leip compress --input_path workspace/models/lenet_gtc/low_precision --quantizer asymmetric --bits 8 --output_path lenet_lp_evaluate_variants/checkpointCompressed/
```
### LRE FP32 (baseline)
```bash
mkdir lenet_lp_evaluate_variants/compiled_lre_fp32
leip compile --input_path workspace/models/lenet_gtc/low_precision --output_path lenet_lp_evaluate_variants/compiled_lre_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path lenet_lp_evaluate_variants/compiled_lre_fp32/ --framework lre --input_types=float32 --input_path lenet_lp_evaluate_variants/compiled_lre_fp32/bin --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### LRE FP32 (storage)
```bash
mkdir lenet_lp_evaluate_variants/compiled_lre_int8
leip compile --input_path workspace/models/lenet_gtc/low_precision --output_path lenet_lp_evaluate_variants/compiled_lre_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path lenet_lp_evaluate_variants/compiled_lre_int8/ --framework lre --input_types=uint8 --input_path lenet_lp_evaluate_variants/compiled_lre_int8/bin --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt
```
### Convert model to integer
```bash
echo workspace/datasets/mnist/eval/images/seven/5576.jpg > rep_dataset.txt
mkdir lenet_lp_evaluate_variants/tfliteOutput
leip convert --input_path workspace/models/lenet_gtc/low_precision --framework tflite --output_path lenet_lp_evaluate_variants/tfliteOutput --data_type int8 --policy TfLite --rep_dataset rep_dataset.txt
```
### LRE Int8 (full)
```bash
leip compile --input_path lenet_lp_evaluate_variants/tfliteOutput/model_save/inference_model.cast.tflite --output_path lenet_lp_evaluate_variants/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path lenet_lp_evaluate_variants/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path lenet_lp_evaluate_variants/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/mnist/eval/index.txt --class_names workspace/datasets/mnist/eval/class_names.txt --preprocessor rgbtogray_int8
```
