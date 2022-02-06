import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x) # 구체적 계산은 forward method에서 진행
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError # 상속을 구현해야 한다는 사실을 알려줌


class Square(Function):
    def forward(self, x):
        return x**2


if __name__ == "__main__":
    x = Variable(np.array(10))
    f = Square()
    y = f(x)

    print(type(y))
    print(y.data)