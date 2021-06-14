import os
import time
from functools import wraps

def colorstr(*input):
    '''
    输入一串字符和格式，用欧逗号隔开。默认是蓝色加粗。
    eg. 
    colorstr('helloworld', 'blue')
    '''
    string, *args = input if len(input) > 1 else (input[0], 'blue', 'bold')  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def fun_run_time(func):
    '''
    装饰器，用于获取函数的执行时间
    ''' 
    @wraps(func)
    def _inner(*args, **kwargs):
        s_time = time.time()
        ret = func(*args, **kwargs)
        e_time = time.time()
        #
        print(colorstr("\t----function [{}] costs {} s".format(func.__name__, e_time-s_time), 'yellow'))
        return ret
    return _inner

def tic():
    '''
    开始计时。
    t = tic()
    '''
    s_time = time.time()
    return s_time

def toc(s_time, word='tic-toc', number = 1):
    '''
    结束计时。
    toc(t)
    '''
    e_time = time.time()
    print(colorstr(f"\t----module [{word}] costs {(e_time-s_time)} s, for {number} actions, ({(e_time-s_time)/number*1000}ms/action)", 'yellow'))
