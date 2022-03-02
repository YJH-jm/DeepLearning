import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않았습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None: # y.grad = np.array(1.0) 생략 가능 
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad) 

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x): # input이 0차원 ndarray인 경우에는 연산 결과가 np.float 발생 가능 
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        self.input = input 
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # y가 scaler인 경우 ndarray 형태로 변환
        output.set_creator(self) 
        self.output = output
        return output

    def forward(self, x):
        return NotImplementedError

    def backward(self, gy):
        return NotImplementedError

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x): # 파이썬 함수로 구현하여 사용이 쉽도록 만듦
    f = Square()
    return f(x)

def exp(x): # 파이썬 함수로 구현하여 사용이 쉽도록 만듦
    return Exp()(x)


def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

def f(x):
    A = Square()
    B = Exp()
    C = Square()

    return C(B(A(x)))

if __name__ == "__main__":
    x = Variable(np.array(0.5))  
    # x = Variable(0.5) TypeError: <class 'float'>은(는) 지원하지 않았습니다.

    # a = square(x)
    # b = exp(a)
    # y = square(b)
    y = square(exp(square(x))) 

    # y.grad = np.array(1.0)  
    y.backward()
    print(x.grad)
    