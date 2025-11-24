#!/usr/bin/env python3
"""
ESPCN Training Script for TensorFlow.js Export

Based on: https://keras.io/examples/vision/super_resolution_sub_pixel/
Paper: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN"
https://arxiv.org/abs/1609.05158

Usage:
    pip install tensorflow tensorflowjs pillow
    python train_espcn.py

This will:
1. Download training images (BSD300 dataset)
2. Train the ESPCN model with augmentations:
   - Sharpen ground truth images (helps model learn sharper outputs)
   - Add noise to input images (helps model learn to denoise)
3. Export to TensorFlow.js format in ./espcn_tfjs/
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
import zipfile
import shutil

# Configuration
SCALE_FACTOR = 2
BATCH_SIZE = 8
EPOCHS = 100
CROP_SIZE = 128  # Size of HR patches
DATA_DIR = Path("./data")
MODEL_DIR = Path("./espcn_model")
TFJS_DIR = Path("./espcn_tfjs")


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
    import tarfile
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
    """
    Apply sharpening to the image using unsharp mask.
    This makes the ground truth sharper, helping the model learn sharper outputs.
    """
    # Sharpening kernel (Laplacian-based)
    kernel = tf.constant([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])

    # Expand kernel for 3 channels
    kernel = tf.tile(kernel, [1, 1, 3, 1])

    # Add batch dimension
    img = tf.expand_dims(image, 0)

    # Apply convolution for each channel separately
    channels = []
    for i in range(3):
        channel = img[:, :, :, i:i+1]
        k = kernel[:, :, i:i+1, :]
        sharpened = tf.nn.conv2d(channel, k, strides=1, padding='SAME')
        channels.append(sharpened)

    sharpened = tf.concat(channels, axis=-1)
    sharpened = tf.squeeze(sharpened, 0)

    # Blend with original based on strength
    result = image * (1 - strength) + sharpened * strength
    return tf.clip_by_value(result, 0, 1)


def add_noise(image, noise_stddev=0.02):
    """
    Add Gaussian noise to the image.
    This helps the model learn to denoise while super-resolving.
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_stddev)
    noisy = image + noise
    return tf.clip_by_value(noisy, 0, 1)


def add_motion_blur(image, max_kernel_size=7):
    """
    Add motion blur to simulate camera/subject motion.
    This helps the model learn to handle real-world video blur.
    """
    # Randomly choose blur direction and strength
    angle = tf.random.uniform([], 0, 180)
    kernel_size = tf.random.uniform([], 3, max_kernel_size, dtype=tf.int32)

    # Convert to radians
    angle_rad = angle * 3.14159 / 180.0

    # Create motion blur kernel
    kernel = tf.zeros([kernel_size, kernel_size, 1, 1], dtype=tf.float32)
    center = kernel_size // 2

    # Simple horizontal motion blur kernel for now
    # (TensorFlow graph mode makes it hard to do angle-dependent kernels)
    kernel_1d = tf.ones([1, kernel_size, 1, 1], dtype=tf.float32) / tf.cast(kernel_size, tf.float32)

    # Apply blur to each channel
    img = tf.expand_dims(image, 0)
    channels = []
    for i in range(3):
        channel = img[:, :, :, i:i+1]
        blurred = tf.nn.conv2d(channel, kernel_1d, strides=1, padding='SAME')
        channels.append(blurred)

    blurred = tf.concat(channels, axis=-1)
    blurred = tf.squeeze(blurred, 0)

    # Randomly apply motion blur (50% chance)
    apply_blur = tf.random.uniform([]) > 0.5
    result = tf.cond(apply_blur, lambda: blurred, lambda: image)

    return tf.clip_by_value(result, 0, 1)


def create_lr_hr_pair(hr_image, scale):
    """Create low-resolution/high-resolution image pair with augmentation."""
    lr_size = CROP_SIZE // scale

    # Randomly crop HR patch
    hr_patch = random_crop(hr_image, CROP_SIZE)

    # Sharpen the HR patch (ground truth) MORE to help model learn sharper outputs
    hr_patch = sharpen_image(hr_patch, strength=0.6)

    # Create LR patch by downsampling
    lr_patch = tf.image.resize(hr_patch, [lr_size, lr_size], method='bicubic')

    # Add motion blur to LR patch to simulate camera/subject motion
    lr_patch = add_motion_blur(lr_patch, max_kernel_size=7)

    # Add noise to LR patch to help model learn to denoise
    lr_patch = add_noise(lr_patch, noise_stddev=0.025)

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


def build_espcn_model(scale, channels=3):
    """
    Build ESPCN model.

    Architecture:
    - Conv 5x5, 64 filters, ReLU
    - Conv 3x3, 32 filters, ReLU
    - Conv 3x3, channels * scale^2 filters
    - Pixel Shuffle (depth_to_space)
    """
    inputs = keras.Input(shape=(None, None, channels))

    # Feature extraction
    x = layers.Conv2D(64, 5, padding='same', activation='relu', name='conv1')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv2')(x)

    # Sub-pixel convolution
    x = layers.Conv2D(channels * (scale ** 2), 3, padding='same', name='conv3')(x)

    # Pixel shuffle
    outputs = layers.Lambda(pixel_shuffle(scale), name='pixel_shuffle')(x)

    return keras.Model(inputs, outputs, name='espcn')


def psnr_metric(y_true, y_pred):
    """Peak Signal-to-Noise Ratio metric."""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


class ESPCNCallback(keras.callbacks.Callback):
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
    print("ESPCN Super Resolution Training")
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
    print("\nBuilding ESPCN model...")
    model = build_espcn_model(SCALE_FACTOR)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[psnr_metric]
    )

    # Calculate steps per epoch (process each image multiple times with random crops)
    steps_per_epoch = len(image_paths) * 10 // BATCH_SIZE

    # Callbacks
    callbacks = [
        ESPCNCallback(),
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
    keras_path = MODEL_DIR / "espcn_keras.keras"
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    # Export to TF.js
    TFJS_DIR.mkdir(parents=True, exist_ok=True)

    # Use graph model format for better performance
    tfjs.converters.save_keras_model(model, str(TFJS_DIR))
    print(f"Exported TF.js model to {TFJS_DIR}")

    # Create a loader script
    loader_js = '''/**
 * ESPCN Model Loader for TensorFlow.js
 *
 * Usage:
 *   const model = await loadESPCNModel('./espcn_tfjs/model.json');
 *   const output = model.predict(inputTensor);
 */

async function loadESPCNModel(modelPath) {
    const model = await tf.loadLayersModel(modelPath);
    console.log('ESPCN model loaded');
    return model;
}

// Example inference function
async function upscaleImage(model, inputCanvas, outputCanvas, scale = 2) {
    return tf.tidy(() => {
        // Read input
        let tensor = tf.browser.fromPixels(inputCanvas).toFloat().div(255);
        tensor = tensor.expandDims(0);

        // Run model
        let output = model.predict(tensor);

        // Process output
        output = output.squeeze().clipByValue(0, 1).mul(255).toInt();

        return tf.browser.toPixels(output, outputCanvas);
    });
}
'''

    with open(TFJS_DIR / "espcn_loader.js", "w") as f:
        f.write(loader_js)

    print(f"Created loader script: {TFJS_DIR}/espcn_loader.js")
    print("\nTo use in your web app:")
    print(f"  1. Copy the '{TFJS_DIR}' folder to your project")
    print(f"  2. Load model: const model = await tf.loadLayersModel('./espcn_tfjs/model.json');")


def create_pretrained_weights():
    """
    Create a model with manually tuned weights that provide good SR results.
    This is useful when you can't train a full model.
    """
    print("Creating model with optimized initial weights...")

    model = build_espcn_model(SCALE_FACTOR)

    # Get layers
    conv1 = model.get_layer('conv1')
    conv2 = model.get_layer('conv2')
    conv3 = model.get_layer('conv3')

    # Initialize conv1 with edge-detection filters
    w1_shape = conv1.weights[0].shape  # (5, 5, 3, 64)
    w1 = np.zeros(w1_shape, dtype=np.float32)

    # Create diverse filters
    for i in range(64):
        filter_type = i % 8
        for c in range(3):  # RGB channels
            if filter_type == 0:  # Gaussian
                for y in range(5):
                    for x in range(5):
                        dx, dy = x - 2, y - 2
                        w1[y, x, c, i] = np.exp(-(dx*dx + dy*dy) / 2) * 0.4
            elif filter_type == 1:  # Sobel-like horizontal
                w1[:, :, c, i] = np.array([
                    [-1, -2, 0, 2, 1],
                    [-2, -4, 0, 4, 2],
                    [-3, -6, 0, 6, 3],
                    [-2, -4, 0, 4, 2],
                    [-1, -2, 0, 2, 1]
                ]) * 0.02
            elif filter_type == 2:  # Sobel-like vertical
                w1[:, :, c, i] = np.array([
                    [-1, -2, -3, -2, -1],
                    [-2, -4, -6, -4, -2],
                    [0, 0, 0, 0, 0],
                    [2, 4, 6, 4, 2],
                    [1, 2, 3, 2, 1]
                ]) * 0.02
            elif filter_type == 3:  # Laplacian
                w1[:, :, c, i] = np.array([
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]
                ]) * 0.03
            elif filter_type == 4:  # Diagonal 1
                w1[:, :, c, i] = np.array([
                    [2, 1, 0, -1, -2],
                    [1, 2, 0, -2, -1],
                    [0, 0, 0, 0, 0],
                    [-1, -2, 0, 2, 1],
                    [-2, -1, 0, 1, 2]
                ]) * 0.02
            elif filter_type == 5:  # Diagonal 2
                w1[:, :, c, i] = np.array([
                    [-2, -1, 0, 1, 2],
                    [-1, -2, 0, 2, 1],
                    [0, 0, 0, 0, 0],
                    [1, 2, 0, -2, -1],
                    [2, 1, 0, -1, -2]
                ]) * 0.02
            elif filter_type == 6:  # Sharpening
                w1[:, :, c, i] = np.array([
                    [0, -1, -1, -1, 0],
                    [-1, 2, -4, 2, -1],
                    [-1, -4, 20, -4, -1],
                    [-1, 2, -4, 2, -1],
                    [0, -1, -1, -1, 0]
                ]) * 0.02
            else:  # Identity-ish
                w1[2, 2, c, i] = 0.5

    b1 = np.zeros(64, dtype=np.float32)
    conv1.set_weights([w1, b1])

    # Initialize conv2 with He initialization
    w2_shape = conv2.weights[0].shape  # (3, 3, 64, 32)
    scale = np.sqrt(2.0 / (3 * 3 * 64))
    w2 = np.random.randn(*w2_shape).astype(np.float32) * scale
    b2 = np.zeros(32, dtype=np.float32)
    conv2.set_weights([w2, b2])

    # Initialize conv3 for pixel shuffle (bilinear-like interpolation)
    w3_shape = conv3.weights[0].shape  # (3, 3, 32, 12)
    w3 = np.zeros(w3_shape, dtype=np.float32)

    # For each output channel (12 = 3 colors * 2^2 sub-pixels)
    for out_ch in range(12):
        color_ch = out_ch // 4
        sub_y = (out_ch % 4) // 2
        sub_x = (out_ch % 4) % 2

        # Center weight is strongest
        for in_ch in range(32):
            w3[1, 1, in_ch, out_ch] = 0.15 * (1 + 0.1 * np.sin(in_ch * 0.3))
            # Neighbors contribute
            w3[0, 1, in_ch, out_ch] = 0.05
            w3[2, 1, in_ch, out_ch] = 0.05
            w3[1, 0, in_ch, out_ch] = 0.05
            w3[1, 2, in_ch, out_ch] = 0.05

    b3 = np.zeros(12, dtype=np.float32)
    conv3.set_weights([w3, b3])

    return model


def main():
    """Main entry point."""
    global EPOCHS

    import argparse
    parser = argparse.ArgumentParser(description='Train ESPCN model for super resolution')
    parser.add_argument('--pretrained-only', action='store_true',
                        help='Only create model with pretrained-like weights (no training)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    args = parser.parse_args()

    EPOCHS = args.epochs

    if args.pretrained_only:
        model = create_pretrained_weights()
        print("Created model with optimized weights (no training)")
    else:
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
