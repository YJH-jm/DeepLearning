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
        if self.grad is None: 
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad) 

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 임의 개수의 인수 (가변 길이 인수)를 건네 받아 호출 가능, 가변 인수는 tuple 형식으로 전달
        # print("확인 : ", type(inputs)) # <class 'tuple'>
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 리스트의 원소를 낱개로 풀어서 전송

        if  not isinstance(ys, tuple): # 결과가 튜플이 아닌 경우
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs 
        self.outputs = outputs
        
        return outputs if len(outputs) > 1 else  outputs[0] # list의 원소가 하나라면 리스트 말고 그 원소만 반환

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


class Add(Function):
    def forward(self, x0, x1):
        # x0, x1 = xs # xs 변수 2개 담긴 list 
        y = x0 + x1
        return y 

def square(x): 
    f = Square()
    return f(x)

def exp(x): 
    return Exp()(x)


def add(x0, x1): # add 함수 구현 
    return Add()(x0,x1)


if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))

    f = Add()

    y = f(x0, x1)

    print(y.data)