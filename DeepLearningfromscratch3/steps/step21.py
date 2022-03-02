import numpy as np
import weakref
import contextlib # with 문이 수반되는 일반적인 작업을 위한 유틸리티를 제공하는 모듈


class Config:
    enable_backprop = True


@contextlib.contextmanager # 직접 작성한 함수를 with문에서 쉽게 사용
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield # yield 이후 전처리 logic, yield 이후 후처리 logic 작성
        # with block 안으로 들어갈 때 전처리가 실행, 빠져나오고 후처리가 실행
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:

    __array_priority__ = 200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않았습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0 # 세대를 입력하는 변수  

    @property
    def shape(self):
        return self.data.shape
        
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None: 
            self.grad = np.ones_like(self.data)


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


    def __len__(self):
        return len(self.data)

    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    # def __mul__(self, other):
    #     return mul(self, other)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 임의 개수의 인수 (가변 길이 인수)를 건네 받아 호출 가능 
        # print("확인 : ", type(inputs)) # <class 'tuple'>
        inputs = [as_variable(x) for x in inputs]
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
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

