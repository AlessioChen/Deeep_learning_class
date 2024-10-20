import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

# Obtain data from Keras API
def get_keras_mlp_data(dataset):
    """Get flattened data using the Keras API (for MLP)

    Parameters
    ----------
    dataset : str
        Name of the dataset to be loaded (mnist, fashion_mnist, cifar10)

    Raises
    ------
    ValueError
        If the dataset is unknown

    """
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # Data is a numpy array of shape (N,W,H) for BW, or (N,W,H,C) for color, and
    # has to be flattened before feeding an MLP

    input_dim = np.prod(X_train.shape[1:])  # i.e. 28x28
    X_train = X_train.reshape((-1,input_dim))
    X_test = X_test.reshape((-1,input_dim))

    # Normalize or standardize the data
    X_train = X_train/255.
    X_test = X_test/255.
    # mu = np.mean(X_train, axis=0)
    # sigma = np.std(X_train, axis=0)
    # epsilon = 1e-6  # Some pixels might be always zero, so std might be zero
    # X_train = (X_train-mu)/np.sqrt(np.square(sigma)+epsilon)
    # X_test = (X_test-mu)/np.sqrt(np.square(sigma)+epsilon)

    # Create-one-hot targets for softmax
    num_classes = len(np.unique(y_train))
    EYE = np.eye(num_classes)
    y_train_oh = EYE[y_train]
    y_test_oh = EYE[y_test]
    return {'train':(X_train, y_train_oh),
            'test':(X_test, y_test_oh),
            'num_classes': num_classes,
            'input_dim': input_dim
            }


if __name__ == '__main__':
    data = get_keras_mlp_data('cifar10')
