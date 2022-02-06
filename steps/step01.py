import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data

if __name__ == "__main__":

    # data = np.array(1.0)
    data = np.array(2.0)
    print(data.shape, data.dtype)
    x = Variable(data)
    x.data = np.array(3.0)
    print(x.data)