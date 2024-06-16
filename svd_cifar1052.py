from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def visualize_sigma_energy(sigma: np.ndarray, threshold: float = 0.9) -> None:
    """Visualize the energy of the singular values

    Args:
        sigma (np.ndarray): Singular values
        threshold (float, optional): Energy threshold. Defaults to 0.9.
    """
    total_energy = np.sum(sigma)
    # total_energy is the sum of all the singular values
    energy = np.cumsum(sigma) / total_energy
    # energy is a list of the cumulative sum of the singular values divided by the total energy
    k = np.argmax(energy > threshold)
    # give the index of the first value in energy that is greater than the threshold
    print(f"Number of components to retain {k}")
    # plt.plot(energy)
    # plt.axvline(k, color="red", linestyle="--")
    # plt.axhline(energy[k], color="red", linestyle="--")
    # plt.xlabel("Number of components")
    # plt.ylabel("Energy")
    # plt.title("Singular Value Energy")
    # plt.show()


def perform_svd(matrix, k=50):
    # Perform SVD
    matrix = matrix.T
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    visualize_sigma_energy(S)
    

    print("U shape:", U.shape)
    print("S shape:", np.diag(S).shape)
    print("VT shape:", VT.shape)

    # Truncate U and VT to retain only the first k principal components
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    print("U_k shape:", U_k.shape)
    print("S_k shape:", S_k.shape)
    print("VT_k shape:", VT_k.shape)

    

    return U_k, S_k, VT_k



def perform_projection(U_k, matrix):
    flat_test_X = matrix.reshape(matrix.shape[0], -1)
    print("flat_test_X shape:", flat_test_X.shape)
    flat_test_X = flat_test_X.T
    print("U_k shape:", U_k.shape)
    Ut_k = U_k.T
    print("U_k.T shape:", Ut_k.shape)
    X_test = Ut_k @ flat_test_X
    print("U_test k shape:", X_test.shape)
    return X_test


# Load CIFAR-10 dataset
(X_train, train_y), (X_test, test_y) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))

# Flatten the data
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("X_train_mean_flat shape: " + str(X_train_flat.shape), end=" ")
print("----  50000 samples, each with 3072 features")
print("X_test_mean_flat shape: " + str(X_test_flat.shape), end=" ")
print("---- 10000 samples, each with 3072 features")

# Perform SVD on the training data
U_k, S_k, VT_k = perform_svd(X_train_flat, k=256)
X_train = S_k @ VT_k
X_train = X_train.T

print("Shapes of SVD results:")
print("VT_k_train shape: {VT_k_train.shape}")
print(f"X_train shape: {X_train.shape}", end=" ")
print("---- 50000 samples, each with 60 features")

X_test = perform_projection(U_k, X_test_flat)
X_test = X_test.T

print("X_test shape:", X_test.shape, end=" ")
print("---- 10000 samples, each with 60 features")

# one-hot encode the labels
one_hot_train_y = tf.keras.utils.to_categorical(train_y)
one_hot_test_y = tf.keras.utils.to_categorical(test_y)

def create_model(input_shape):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

input_shape = (X_train.shape[1],)
print("input_shape:", input_shape)
model = create_model(input_shape)

es = EarlyStopping(patience=10, monitor="loss")
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=3, verbose=1, min_lr=1e-6
)

history = model.fit(
    X_train,
    one_hot_train_y,
    validation_split=0.20,
    batch_size=64,
    epochs=20,
    shuffle=True,
    callbacks=[es, lr_scheduler],
)

_, acc = model.evaluate(X_test, one_hot_test_y)
print(f"Accuracy: {acc}")
