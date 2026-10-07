import argparse
from pathlib import Path

import tensorflow as tf

from train import parse_function, tfrecord_files


def evaluate(model_path: str | Path, tfrecord_path: str | Path) -> None:
    model_file = Path(model_path).expanduser()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    dataset = (
        tf.data.TFRecordDataset(tfrecord_files(tfrecord_path))
        .map(parse_function)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    model = tf.keras.models.load_model(model_file)

    total = 0
    correct = 0
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0).reshape(-1)
        for probability, actual_label in zip(
            probabilities, labels.numpy().tolist()
        ):
            predicted_label = int(probability >= 0.5)
            actual_label = int(actual_label)
            predicted_text = (
                "Degenerative Spine" if predicted_label else "Normal Spine"
            )
            actual_text = (
                "Degenerative Spine" if actual_label else "Normal Spine"
            )
            print(
                f"Predicted: {predicted_text} "
                f"(probability: {probability:.4f}), actual: {actual_text}"
            )
            total += 1
            correct += predicted_label == actual_label

    if total == 0:
        raise ValueError(f"No examples found in: {tfrecord_path}")
    accuracy = correct / total * 100
    print(
        f"Accuracy on supplied TFRecords: {accuracy:.2f}% "
        f"({correct}/{total})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved classifier on labeled TFRecord examples."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a saved Keras model.",
    )
    parser.add_argument(
        "--tfrecords",
        required=True,
        help="TFRecord file or directory containing .tfrecord files.",
    )
    args = parser.parse_args()
    evaluate(args.model, args.tfrecords)


if __name__ == "__main__":
    main()
