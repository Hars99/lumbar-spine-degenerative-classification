import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


FEATURE_DESCRIPTION = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64),
    "image/width": tf.io.FixedLenFeature([], tf.int64),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}

DATA_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)


def tfrecord_files(path: str | Path) -> list[str]:
    dataset_path = Path(path).expanduser()
    if dataset_path.is_file():
        return [str(dataset_path)]
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"TFRecord path does not exist: {dataset_path}")

    files = sorted(dataset_path.glob("*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files found in: {dataset_path}")
    return [str(file) for file in files]


def parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(
        example_proto, FEATURE_DESCRIPTION
    )
    image = tf.io.decode_jpeg(parsed_features["image/encoded"], channels=3)
    image = tf.image.resize(image, [224, 224])
    labels = tf.sparse.to_dense(parsed_features["image/object/class/label"])
    label = tf.reduce_any(tf.equal(labels, 1))
    return image, tf.cast(label, tf.int32)


def parse_function_with_augmentation(example_proto):
    image, label = parse_function(example_proto)
    return DATA_AUGMENTATION(image), label


def balanced_augmentation(tfrecord_path: str | Path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files(tfrecord_path))

    def filter_normal(example_proto):
        _, label = parse_function(example_proto)
        return tf.equal(label, 0)

    def filter_degenerative(example_proto):
        _, label = parse_function(example_proto)
        return tf.equal(label, 1)

    normal = raw_dataset.filter(filter_normal).repeat(3)
    degenerative = raw_dataset.filter(filter_degenerative)
    balanced_dataset = normal.concatenate(degenerative).shuffle(1000)
    return (
        balanced_dataset.map(parse_function_with_augmentation)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_dataset(tfrecord_path: str | Path, augment: bool = False):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files(tfrecord_path))
    parser = parse_function_with_augmentation if augment else parse_function
    return (
        raw_dataset.map(parser)
        .shuffle(1000)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )


def create_custom_cnn(input_shape):
    cnn_input = Input(shape=input_shape, name="cnn_input")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(cnn_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=cnn_input, outputs=x, name="Custom_CNN")


def create_custom_model():
    model_input = Input(shape=(224, 224, 3), name="model_input")
    densenet_base = DenseNet121(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    densenet_base.trainable = True
    for layer in densenet_base.layers[:100]:
        layer.trainable = False

    densenet_output = GlobalAveragePooling2D(name="densenet_gap")(
        densenet_base.output
    )
    cnn_output = create_custom_cnn(input_shape=(224, 224, 3))(model_input)
    combined = Concatenate(name="concat_features")(
        [densenet_output, cnn_output]
    )
    dense_layer = Dense(256, activation="relu", name="dense_layer")(combined)
    dense_layer = Dropout(0.5)(dense_layer)
    output = Dense(1, activation="sigmoid", name="output_layer")(dense_layer)
    return Model(
        inputs=model_input,
        outputs=output,
        name="Hybrid_DenseNet_CNN_Model",
    )


def train_model(
    train_dir: str | Path,
    validation_dir: str | Path,
    output_model: str | Path,
    epochs: int,
) -> None:
    train_dataset = balanced_augmentation(train_dir)
    valid_dataset = load_dataset(validation_dir)

    labels = []
    for _, batch_labels in load_dataset(train_dir):
        labels.extend(batch_labels.numpy().tolist())
    if not labels:
        raise ValueError(f"No training examples found in: {train_dir}")

    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, class_weights)
    }
    print("Class weights:", class_weights)

    model = create_custom_model()
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        class_weight=class_weights,
    )

    _, val_accuracy, val_recall = model.evaluate(valid_dataset, verbose=1)
    print(
        f"Validation accuracy: {val_accuracy * 100:.2f}%, "
        f"validation recall: {val_recall * 100:.2f}%"
    )
    output_path = Path(output_model).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the hybrid DenseNet121 and custom CNN classifier."
    )
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Directory with training TFRecords.",
    )
    parser.add_argument(
        "--validation-dir",
        required=True,
        help="Directory with validation TFRecords.",
    )
    parser.add_argument(
        "--output-model",
        default="binary_classification_model.h5",
        help="Output path for the trained Keras model.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be greater than zero.")

    train_model(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        output_model=args.output_model,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
