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
            
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx

                else:
                    x.grad = x.grad + gx                
                
                if x.creator is not None:
                    funcs.append(x.creator)

    
    def cleargrad(self): # 변수 재사용시 
        self.grad = None


def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 임의 개수의 인수 (가변 길이 인수)를 건네 받아 호출 가능 
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
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x): 
    f = Square()
    return f(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x): 
    return Exp()(x)

class Add(Function):
    def forward(self, x0, x1):
        # x0, x1 = xs # xs 변수 2개 담긴 list 
        y = x0 + x1
        return y 
    
    def backward(self,gy):
        return gy, gy

def add(x0, x1): # add 함수 구현 
    return Add()(x0,x1)


if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()

    y = add(add(x, x), x)
    y.backward()
    print(x.grad)