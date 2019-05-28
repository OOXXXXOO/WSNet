import torch.nn as nn
import torch
import matplotlib.pyplot as plt





def main():
    tensor=torch.rand(100,100)
    print(tensor)
    plt.imshow(tensor.numpy()),plt.show()


if __name__ == '__main__':
    main()
    