#!/usr/bin/env python3
"""
MobileNetV2-based Super Resolution Training Script

Uses MobileNetV2 as feature extractor for better quality super resolution.
MobileNet is efficient and produces better results than simple ESPCN.

Usage:
    pip install tensorflow tensorflowjs pillow
    python train_mobilenet_sr.py

This will:
1. Download training images (BSD300 dataset)
2. Train MobileNetV2-based SR model with augmentations:
   - Sharpen ground truth images (helps model learn sharper outputs)
   - Add light noise to input images (helps model learn to denoise)
   - Add motion blur (helps handle video blur)
3. Export to TensorFlow.js format in ./mobilenet_sr_tfjs/
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs
from pathlib import Path
import urllib.request
import tarfile

# Configuration
SCALE_FACTOR = 2
BATCH_SIZE = 8
EPOCHS = 100
CROP_SIZE = 128  # Size of HR patches
DATA_DIR = Path("./data")
MODEL_DIR = Path("./mobilenet_sr_model")
TFJS_DIR = Path("./mobilenet_sr_tfjs")


def download_bsd300():
    """Download BSD300 dataset for training."""
    url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    data_dir = DATA_DIR / "BSDS300"

    if data_dir.exists():
        print("BSD300 dataset already downloaded")
        return data_dir / "images" / "train"

    print("Downloading BSD300 dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tgz_path = DATA_DIR / "BSDS300-images.tgz"
    urllib.request.urlretrieve(url, tgz_path)

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)

    os.remove(tgz_path)
    print("Dataset ready")
    return data_dir / "images" / "train"


def load_image(path):
    """Load and preprocess an image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def random_crop(image, crop_size):
    """Random crop for data augmentation."""
    return tf.image.random_crop(image, [crop_size, crop_size, 3])


def sharpen_image(image, strength=1.0):
    """Apply sharpening to the image."""
    kernel = tf.constant([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    img = tf.expand_dims(image, 0)
    channels = []
    for i in range(3):
        channel = img[:, :, :, i:i+1]
        k = kernel[:, :, i:i+1, :]
        sharpened = tf.nn.conv2d(channel, k, strides=1, padding='SAME')
        channels.append(sharpened)

    sharpened = tf.concat(channels, axis=-1)
    sharpened = tf.squeeze(sharpened, 0)

    result = image * (1 - strength) + sharpened * strength
    return tf.clip_by_value(result, 0, 1)


def add_noise(image, noise_stddev=0.01):
    """Add light Gaussian noise."""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev)
    noisy = image + noise
    return tf.clip_by_value(noisy, 0, 1)


def add_motion_blur(image, max_kernel_size=7):
    """Add motion blur to simulate video blur."""
    kernel_size = tf.random.uniform([], 3, max_kernel_size, dtype=tf.int32)
    kernel_1d = tf.ones([1, kernel_size, 1, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)

    img = tf.expand_dims(image, 0)
    channels = []
    for i in range(3):
        channel = img[:, :, :, i:i+1]
        blurred = tf.nn.conv2d(channel, kernel_1d, strides=1, padding='SAME')
        channels.append(blurred)

    blurred = tf.concat(channels, axis=-1)
    blurred = tf.squeeze(blurred, 0)

    # Randomly apply motion blur (30% chance)
    apply_blur = tf.random.uniform([]) > 0.7
    result = tf.cond(apply_blur, lambda: blurred, lambda: image)

    return tf.clip_by_value(result, 0, 1)


def create_lr_hr_pair(hr_image, scale):
    """Create low-resolution/high-resolution image pair with augmentation."""
    lr_size = CROP_SIZE // scale

    # Randomly crop HR patch
    hr_patch = random_crop(hr_image, CROP_SIZE)

    # Sharpen the HR patch (ground truth)
    hr_patch = sharpen_image(hr_patch, strength=0.5)

    # Create LR patch by downsampling
    lr_patch = tf.image.resize(hr_patch, [lr_size, lr_size], method='bicubic')

    # Add motion blur occasionally
    lr_patch = add_motion_blur(lr_patch, max_kernel_size=5)

    # Add light noise (reduced from 0.025 to 0.01)
    lr_patch = add_noise(lr_patch, noise_stddev=0.01)

    lr_patch = tf.clip_by_value(lr_patch, 0, 1)
    hr_patch = tf.clip_by_value(hr_patch, 0, 1)

    return lr_patch, hr_patch


def create_dataset(image_paths, batch_size, scale):
    """Create TF dataset for training."""
    def process_path(path):
        img = load_image(path)
        return create_lr_hr_pair(img, scale)

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()

    return dataset


def pixel_shuffle(scale):
    """Sub-pixel convolution layer (depth to space)."""
    return lambda x: tf.nn.depth_to_space(x, scale)


def build_mobilenet_sr_model(scale, channels=3):
    """
    Build MobileNetV2-based SR model.

    Uses MobileNetV2 as feature extractor followed by upsampling layers.
    Much more powerful than simple ESPCN.
    """
    inputs = keras.Input(shape=(None, None, channels))

    # Use MobileNetV2 as feature extractor (pretrained on ImageNet)
    # Remove top layers, use as feature extractor
    mobilenet = keras.applications.MobileNetV2(
        input_shape=(None, None, channels),
        include_top=False,
        weights=None,  # Start from scratch for SR task
        alpha=0.5  # Width multiplier for smaller model
    )

    # Extract features
    x = mobilenet(inputs)

    # Additional refinement layers
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    # Sub-pixel convolution for upsampling
    x = layers.Conv2D(channels * (scale ** 2), 3, padding='same')(x)
    outputs = layers.Lambda(pixel_shuffle(scale))(x)

    return keras.Model(inputs, outputs, name='mobilenet_sr')


def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio metric."""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


class TrainingCallback(keras.callbacks.Callback):
    """Callback to display training progress."""
    def __init__(self):
        super().__init__()
        self.best_psnr = 0

    def on_epoch_end(self, epoch, logs=None):
        psnr = logs.get('psnr_metric', 0)
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            print(f" - Best PSNR: {psnr:.2f} dB")


def train_model():
    """Main training function."""
    print("=" * 60)
    print("MobileNetV2-based Super Resolution Training")
    print("=" * 60)

    # Download dataset
    train_dir = download_bsd300()

    # Get image paths
    image_paths = list(train_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} training images")

    if len(image_paths) == 0:
        print("No training images found. Please check the dataset.")
        return None

    # Create dataset
    train_dataset = create_dataset(
        [str(p) for p in image_paths],
        BATCH_SIZE,
        SCALE_FACTOR
    )

    # Build model
    print("\nBuilding MobileNetV2-SR model...")
    model = build_mobilenet_sr_model(SCALE_FACTOR)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[psnr_metric]
    )

    # Calculate steps per epoch
    steps_per_epoch = len(image_paths) * 10 // BATCH_SIZE

    # Callbacks
    callbacks = [
        TrainingCallback(),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=20,
            restore_best_weights=True
        )
    ]

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    return model


def export_to_tfjs(model):
    """Export trained model to TensorFlow.js format."""
    print("\n" + "=" * 60)
    print("Exporting to TensorFlow.js")
    print("=" * 60)

    # Save Keras model first
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keras_path = MODEL_DIR / "mobilenet_sr.keras"
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    # Export to TF.js
    TFJS_DIR.mkdir(parents=True, exist_ok=True)

    tfjs.converters.save_keras_model(model, str(TFJS_DIR))
    print(f"Exported TF.js model to {TFJS_DIR}")

    print("\nTo use in your web app:")
    print(f"  1. Copy the '{TFJS_DIR}' folder to your project")
    print(f"  2. Update espcn.js to load: './mobilenet_sr_tfjs/model.json'")


def main():
    """Main entry point."""
    global EPOCHS

    import argparse
    parser = argparse.ArgumentParser(description='Train MobileNetV2-SR model')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    args = parser.parse_args()

    EPOCHS = args.epochs

    model = train_model()
    if model is None:
        print("Training failed")
        return

    # Export to TF.js
    export_to_tfjs(model)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
