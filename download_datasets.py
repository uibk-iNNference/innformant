from resources import data

if __name__ == "__main__":
    data.get_mnist_data()
    data.get_cifar10_data()
    data.get_fmnist_data()
    data.get_imagenet_data()
