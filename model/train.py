"""
Handwritten Digit Recognition - Baseline Feed-Forward Neural Network Training Script

Step 2: Training a baseline Dense Neural Network (MLP) on the MNIST dataset.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf


def train_baseline_model():
    print("=" * 60)
    print("Starting Step 2: Baseline Model Training on MNIST")
    print("=" * 60)

    # 1. Load MNIST dataset
    print("\n1. Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(f"   Loaded {len(x_train)} training images and {len(x_test)} test images.")

    # 2. Normalize image pixel values from [0, 255] to [0.0, 1.0]
    print("\n2. Normalizing image pixel values to range [0.0, 1.0]...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3. Build feed-forward neural network architecture
    print("\n3. Building Sequential Neural Network Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28), name="input_image"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_hidden_1"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_hidden_2"),
        tf.keras.layers.Dense(10, activation="softmax", name="output_digits")
    ])

    # 4. Print model architecture summary
    print("\n4. Model Summary:")
    model.summary()

    # 5. Compile model with Adam optimizer and sparse_categorical_crossentropy loss
    print("\n5. Compiling model (Optimizer: Adam, Loss: Sparse Categorical Crossentropy)...")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6 & 7. Train model with a validation split
    epochs = 10
    batch_size = 64
    print(f"\n6. Training model for {epochs} epochs with validation split of 10%...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 9 & 10. Evaluate trained model on test dataset
    print("\n7. Evaluating model performance on test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\n   --> Test Loss: {test_loss:.4f}")
    print(f"   --> Test Accuracy: {test_accuracy * 100:.2f}%")

    # 11. Save trained model into model/saved_models/handwritten_digit_model.keras
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    saved_model_path = os.path.join(saved_models_dir, "handwritten_digit_model.keras")
    model.save(saved_model_path)
    print(f"\n8. Saved trained model artifact to: {saved_model_path}")

    # 12 & 13. Plot training and validation metrics
    print("\n9. Plotting and saving training history curves...")
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Model Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss", marker="o")
    plt.plot(history.history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Model Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(saved_models_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   Training history plot saved to: {plot_path}")

    # Final summary output
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    total_params = model.count_params()

    print("\n" + "=" * 60)
    print("STEP 2 TRAINING COMPLETE SUMMARY")
    print("=" * 60)
    print(f"Architecture         : Flatten -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Final Train Accuracy : {final_train_acc * 100:.2f}%")
    print(f"Final Val Accuracy   : {final_val_acc * 100:.2f}%")
    print(f"Final Test Accuracy  : {test_accuracy * 100:.2f}%")
    print(f"Saved Model Location : {saved_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_baseline_model()
