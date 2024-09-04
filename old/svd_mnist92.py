import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

""" Creating a simple neural network model with SVD on mnist data instead of doing convolutions.

    The model is a simple neural network with 2 hidden layers and 1 output layer.
    The input data is transformed using SVD before being fed to the model.
    The training data is transformed using SVD and the test data is transformed using
    
    My training dataset is 60000x28x28 and test dataset is 10000x28x28.
    Note this is 60000 columns of 28x28 images.!!!!
    I reshape the training data to 60000x784 and test data to 10000x784.
    I then perform SVD on the training data to get U, S, Vt.
    
    U is a 60000x784 matrix, S is a 784x784 matrix and Vt is a 784x784 matrix.
    
    Atrain = U * S * Vt
    
    I then take the k rank approximation of U and S to get U_k and S_k.
    U_k is a 60000xk matrix and S_k is a kxk matrix.
    
    rAtrain = U_k * S_k (reduce the dimensions of the data)
    NOTE: U_k * S_k is the same as scaled_train_X * V_k
    
    This is why the projection of the test data is done using V_k.
    rAtest = Atest * V_k
    
    Then I train the model on rAtrain and test it on rAtest.
"""

""" 
Model Architecture:

1. **Input Layer**
2. **Flatten Layer**
3. **Dense Layer 1**: 32 units, ReLU activation
4. **Dense Layer 2**: 16 units, ReLU activation
5. **Dense Layer 3**: 10 units, Softmax activation
6. **Output Layer**
"""


def SteveNet(input_shape: tuple) -> Model:
    """Simple neural network model

    Args:
        input_shape (tuple): shape of the input data

    Returns:
        Model: keras model
    """
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(32, activation="relu")(input_layer)
    x = layers.Dense(16, activation="relu")(x)
    output_layer = layers.Dense(10, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_scaled_data() -> tuple:
    """Get the scaled mnist data

    Returns:
        tuple: scaled_train_X, one_hot_train_y, scaled_test_X, one_hot_test_y
    """
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print("X_train:", train_X.shape)
    print("Y_train:", train_y.shape)
    print("X_test: ", test_X.shape)
    print("Y_test: ", test_y.shape)
    scaled_train_X = train_X / 255.0
    scaled_test_X = test_X / 255.0
    one_hot_train_y = to_categorical(train_y)
    one_hot_test_y = to_categorical(test_y)
    return scaled_train_X, one_hot_train_y, scaled_test_X, one_hot_test_y


def transform_train_svd_data(scaled_train_X: np.ndarray, k: int = 50) -> tuple:
    """Transform the training data using SVD

    Args:
        scaled_train_X (_type_): _description_
        k (int, optional): _description_. Defaults to 50.

    Returns:
        tuple: _description_
    """
    flat_train_X = scaled_train_X.reshape(scaled_train_X.shape[0], -1)
    print("flat_train_X shape:", flat_train_X.shape)
    flat_train_X = flat_train_X.T
    U, S, Vt = np.linalg.svd(flat_train_X, full_matrices=False)
    print("U_train shape:", U.shape)
    print("S_train shape:", S.shape)
    print("Vt_train shape:", Vt.shape)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    print("U_train k shape:", U_k.shape)
    print("Vt_train k shape:", Vt_k.shape)
    print("S_train k shape:", S_k.shape)
    
    return U_k, S_k, Vt_k


def transform_test_data(U_k: np.ndarray, scaled_test_X: np.ndarray) -> np.ndarray:
   
    flat_test_X = scaled_test_X.reshape(scaled_test_X.shape[0], -1)
    print("flat_test_X shape:", flat_test_X.shape)
    flat_test_X = flat_test_X.T
    print("U_k shape:", U_k.shape)
    Ut_k = U_k.T
    print("U_k.T shape:", Ut_k.shape)
    X_test = Ut_k @ flat_test_X
    print("U_test k shape:", X_test.shape)
    return X_test


def main():
    scaled_train_X, one_hot_train_y, scaled_test_X, one_hot_test_y = get_scaled_data()
    
    U_k, S_k, Vt_k = transform_train_svd_data(scaled_train_X, k=10)
    X_train = S_k @ Vt_k
    X_train = X_train.T
    print("X_train shape:", X_train.shape)
    X_test = transform_test_data(U_k, scaled_test_X)
    X_test = X_test.T

    es = EarlyStopping(patience=3, monitor="loss")

    input_shape = (X_train.shape[1],)
    print("input_shape:", input_shape)

    model = SteveNet(input_shape)
    model.summary()

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.002),
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        one_hot_train_y,
        batch_size=32,
        epochs=20,
        callbacks=[es],
    )

    _, acc = model.evaluate(X_test, one_hot_test_y)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()
