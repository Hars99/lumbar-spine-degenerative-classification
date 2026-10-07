# Lumbar Spine Degenerative Classification

Hybrid deep-learning image-classification experiment intended to distinguish normal spine images from degenerative cases using a Roboflow-prepared TFRecord dataset.

> **Research use only.** This repository is an educational ML experiment, not a medical device. Its output must not be used for diagnosis or treatment decisions.

## Overview

The training script reads JPEG images and object-class labels from TFRecord files, resizes each image to 224 × 224, and trains a binary classifier. Its **hybrid model** processes each image through two parallel branches: ImageNet-initialized DenseNet121 and a custom CNN. Their feature vectors are concatenated and passed to a shared binary classification head. This design uses transfer-learned and task-trained feature representations together; the repository does not establish that combining them improves performance. A second script loads a saved Keras model and prints predictions alongside labels from a TFRecord dataset.

The repository contains source code and a sample prediction screenshot, but not the dataset, a trained model, training logs, or an environment lock file.

## Key features

- Parses the Roboflow TFRecord feature schema included in the source.
- Implements a binary label rule over TFRecord object-class IDs; the ID-to-class-name mapping still needs confirmation against the exact export.
- Repeats normal-class training examples three times, applies augmentation, and computes class weights from the training data.
- Combines ImageNet-initialized DenseNet121 features with features from a custom three-convolution CNN.
- Evaluates a saved model against labeled TFRecord examples.

## Dataset Preparation

A major challenge in this project was obtaining a suitable lumbar-spine image dataset for experimentation. The available medical-image datasets were not immediately usable for this project because of differences in image formats, dataset organization, access restrictions, and permission requirements.

Instead of relying on a single ready-made dataset, I created a custom working dataset using **Roboflow**.

My dataset preparation process included:

- gathering images required for the experiment;
- importing and organizing the collected images in Roboflow;
- annotating the images, including bounding-box work;
- preparing the training and testing data;
- managing dataset versions and exports through Roboflow.

Roboflow was therefore used as part of the **data curation and annotation pipeline**, rather than simply as a source for downloading an existing dataset.

Roboflow workspace/project:

`https://app.roboflow.com/dl-ljbpi/augmented_dataset-2/2`

> **Note:** Access to some source medical datasets and images may be subject to their original licenses, permissions, or usage restrictions. Large dataset files are not included directly in this GitHub repository.

The bounding boxes are part of the Roboflow data-preparation workflow. The current TensorFlow code does **not** read box coordinates or train a detector: it reads the image and `image/object/class/label` values to perform image-level classification. Confirm the exported label IDs before interpreting the names printed by the evaluation script.

The checked-in Roboflow metadata notes refer to an export/listing and mention **CC BY 4.0**, but do not establish the rights for every source image or for the exact project/version linked above. Check the original image licenses, access permissions, and export terms before redistributing the dataset. This README does not claim ownership of the source images.

### Dataset Workflow

```text
Image Collection
       ↓
Dataset Review
       ↓
Roboflow Import
       ↓
Annotation / Dataset Organization
       ↓
Train / Test Dataset Preparation
       ↓
TFRecord Export
       ↓
TensorFlow Training Pipeline
       ↓
DenseNet121 + Custom CNN
       ↓
Classification / Evaluation
```

The current repository focuses on the model-training and evaluation pipeline. The training CLI expects training and validation TFRecords; the evaluation CLI can run against a held-out test TFRecord or directory. The dataset itself is not included here.

## Workflow

```text
Roboflow TFRecords (train / valid)
             |
             v
       JPEG decode, resize to 224 x 224, label mapping
             |
             v
 Training: normal-class oversampling + random augmentation
             |
             v
 DenseNet121 branch + custom CNN branch -> sigmoid output
             |
             v
        Keras model file (.h5)
             |
             v
 Evaluation script -> per-image prediction and actual label
```

## Tech stack

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn (`compute_class_weight`)
- Roboflow-exported TFRecord data

The original repository did not record dependency versions, so `requirements.txt` lists the required packages without version pins. Pin and validate versions in the target environment before relying on repeatable training.

## Project structure

```text
.
├── train.py                    # Dataset parsing, model definition, and training
├── evaluate_tfrecord.py        # Load a saved model and evaluate TFRecord examples
├── requirements.txt            # Runtime dependencies
├── README.dataset.txt          # Dataset source and license note
├── README.roboflow.txt         # Roboflow export details
└── image.png                   # Committed sample prediction output
```

Generated model files, local data, and environment files are excluded by `.gitignore`.

## Installation

The commands below use Windows PowerShell:

```powershell
git clone https://github.com/Hars99/lumbar-spine-degenerative-classification.git
cd lumbar-spine-degenerative-classification
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlow may download the ImageNet weights for DenseNet121 the first time the training script initializes the model. That requires network access.

## Dataset

The Roboflow project is [Augmented_dataset](https://app.roboflow.com/dl-ljbpi/augmented_dataset-2/2). Export the prepared data in TensorFlow TFRecord format and place the training and validation TFRecord files in separate directories:

```text
<dataset-root>/
├── train/
│   └── *.tfrecord
└── valid/
    └── *.tfrecord
```

Each directory must contain at least one `.tfrecord` file. Use a held-out test split for the evaluation command. The dataset files are not included in this repository.

## Usage

Train from the repository root, substituting the local dataset paths:

```powershell
python train.py --train-dir "D:\data\Augmented_dataset\train" --validation-dir "D:\data\Augmented_dataset\valid"
```

Training runs for 10 epochs by default and saves `binary_classification_model.h5` in the current directory. Set a different output path with `--output-model` or change the epoch count with `--epochs`:

```powershell
python train.py --train-dir "D:\data\Augmented_dataset\train" --validation-dir "D:\data\Augmented_dataset\valid" --epochs 10 --output-model "models\spine_classifier.h5"
```

Evaluate a saved model against a TFRecord file or a directory containing `.tfrecord` files:

```powershell
python evaluate_tfrecord.py --model "binary_classification_model.h5" --tfrecords "D:\data\Augmented_dataset\valid"
```

The evaluation command prints each predicted class, its probability, the actual dataset label, and the overall accuracy for the supplied examples. It evaluates labeled TFRecords; it does not currently accept arbitrary image files.

## Model and methodology

This is a **hybrid deep-learning architecture**, not an object detector: both neural-network branches receive the same image, and their learned features are fused for image-level classification.

Images are decoded as three-channel JPEGs and resized to 224 × 224. The parser assigns the positive output label when any object class ID in `image/object/class/label` equals `1`; otherwise it assigns the negative label. The repository's metadata names `Spinal-canal-stenosis` as an annotation class, but does not verify its numeric ID for the linked export. Confirm the class-ID mapping before interpreting class names emitted by evaluation. Bounding-box coordinates are not parsed or used by this classifier.

The training pipeline repeats class-0 examples three times, applies horizontal/vertical flips, rotation, and zoom, and calculates class weights from the parsed training labels.

The model has two branches over the same input:

1. **DenseNet121:** ImageNet weights, no classification head, with the first 100 layers frozen; global average pooling produces image features.
2. **Custom CNN:** convolution layers with 32, 64, and 128 filters, with max pooling after the first two layers and global average pooling at the end.

The feature vectors are concatenated, passed through a 256-unit ReLU layer and 0.5 dropout, then classified by a one-unit sigmoid output. Training uses binary cross-entropy, Adam with exponential learning-rate decay, accuracy and recall metrics, and 10 epochs by default.

## Results

The original project README reported 95.67% accuracy and 97.38% validation recall. Treat these as **historical, unverified experiment results**: this repository does not include the exact dataset/split, trained model, or evaluation logs needed to reproduce them. The included `image.png` shows example TFRecord predictions, not aggregate metric evidence.

## Limitations and future work

- The DenseNet branch uses ImageNet-pretrained weights but receives decoded/resized pixel values (nominally 0–255) without DenseNet121's `preprocess_input`. The Keras DenseNet preprocessing converts RGB to BGR and zero-centers channels using ImageNet means; omitting it makes the inputs inconsistent with the pretrained backbone. This is a limitation of the historical experiment. Correcting preprocessing changes model inputs, so retrain and re-evaluate rather than applying it to an existing saved model or comparing directly with the historical metrics.
- Validate augmentation choices for the target anatomy; the current training transform includes vertical flips.
- The repository does not include a fixed random seed, training/evaluation logs, model artifact, or pinned dependency versions.
- Confirm that the dataset was split before augmentation and check for related-image leakage across splits.
- Add automated tests for TFRecord parsing, label mapping, CLI validation, and model loading.
- Add support for single image files only after defining and testing a consistent image preprocessing and output workflow.

## Author

Harshith Manikhanta Sunkara
