#模拟sum函数的注意几点
#1.接收两个参数，及返回值处理
#2.空对象问题处理
#3.传入对象数据类型处理
from collections.abc import Iterable

def imiatteSum(*args,start = 0):
    #如果传入的是可迭代对象，那么args[0]就是这个对象，如果是一序列数字，那么args[0]就不是可迭代对象
    if not isinstance(start,(int,float,complex)):
        raise Exception('start should be numeric')

    if not bool(args):
        return start

    if isinstance(args[0],Iterable) and len(args) == 1:#只能传入一个可迭代对象
        if not bool(args[0]):
            return start

        s = 0
        for i in args[0]:
            s += i
        return s + start

    elif isinstance(args[0],(int,float,complex)):

        s = 0
        for i in args:
            s += i
        return s + start

    else:
        raise Exception('type error')

print(imiatteSum(1,2,3,4))
print(imiatteSum([1,2,3,4]))
#print(imiatteSum([1,2,3,4],[1,2,3]))       #Exception: type error
print(imiatteSum())
#print(imiatteSum(start='12'))              #Exception: start should be numeric

'''sum帮助文档如下
sum(iterable, /, start=0)
    Return the sum of a 'start' value (default: 0) plus an iterable of numbers

    When the iterable is empty, return the start value.
    This function is intended specifically for use with numeric values and may
    reject non-numeric types.
'''
"""判断可迭代对象的方法
from collections import Iterable
isinstance([1,2,3,4], Iterable) ==> True
"""