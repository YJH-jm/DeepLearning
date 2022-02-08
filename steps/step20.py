import numpy as np
import weakref
import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않았습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대를 입력하는 변수  

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)


        # funcs = [self.creator]
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                    # print("확인 : " ,x.grad)

                else:
                    x.grad = x.grad+gx                
                
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None 

    def cleargrad(self):
        self.grad = None

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    # def __mul__(self, other):
    #     return mul(self, other)


def as_array(x): 
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 임의 개수의 인수 (가변 길이 인수)를 건네 받아 호출 가능 
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 리스트의 원소를 낱개로 풀어서 전송

        if  not isinstance(ys, tuple): # 결과가 튜플이 아닌 경우
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]


        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 세대수를 입력


            for output in outputs:
                output.set_creator(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] # 약한 참조로 순환 참조로 인한 메모리 낭비 줄임
        
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
    
    def backward(self,gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0 

def square(x): 
    f = Square()
    return f(x)

def exp(x): 
    return Exp()(x)

def mul(x0, x1):
    return Mul()(x0, x1)

def add(x0, x1): # add 함수 구현 
    return Add()(x0,x1)


Variable.__add__ = add
Variable.__mul__ = mul

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)