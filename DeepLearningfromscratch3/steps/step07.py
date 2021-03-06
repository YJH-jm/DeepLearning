import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None # 이 값이 어떤 함수를 통해서 만들어졌는지 알아내기 위한 변수 

    def self_creator(self, func): 
        self.creator = func

    def backward(self):
        f = self.creator # 함수를 가지고 옴
        if f is not None:
            x = f.input # 함수의 입력을 가지고 옴
            x.grad = f.backward(self.grad) # 함수의 backward 메서드 호출 
            x.backward() # 하나 앞 변수의 backward를 호출

class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.self_creator(self) # 출력 변수에 창조자 설정
        self.output = output # 출력 변수 보관
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
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    # assert y.creator == C
    # assert y.creator.input == b
    # assert y.creator.input.creator == B
    # assert y.creator.input.creator.input == a
    # assert y.creator.input.creator.input.creator == A
    # assert y.creator.input.creator.input.creator.input == x


    # y.grad = np.array(1.0)

    # C = y.creator
    # b = C.input
    # b.grad = C.backward(y.grad)

    # B = b.creator
    # a = B.input
    # a.grad = B.backward(b.grad)

    # A = a.creator
    # x = A.input
    # x.grad = A.backward(a.grad)