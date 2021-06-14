import pickle #小模型快，大模型慢很多
import joblib #大模型也能存，速度都挺快，优选
    

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
from utils.tools import tic, toc, colorstr


def save_obj(obj, path):
    '''
    save_obj(a, 'weights\\2.joblib')
    '''
    joblib.dump(filename=path, value=obj)
    print(colorstr('mode saved.'))
    


def load_obj(path):
    '''
    model1 = load_obj('weights\\2.joblib')
    '''
    print(colorstr('mode loaded.'))
    return joblib.load(filename=path)


















#==============================
def save_obj_pickle(obj, path):
    '''
    保存对象到文件。
    save_obj(a, 'weights\\1.model')
    '''
    file = open(path, 'wb')
    pickle.dump(obj, file)


def load_obj_pickle(path):
    '''
    加载文件到对象。
    b = load_obj('weights\\1.model')
    '''
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

#========================================
if __name__ == "__main__":
    import numpy as np
    
    class m:
        def __init__(self):
            self.name = '123'
            self.a = np.array(range(1))#40000000
            return
        def check(self):
            print(126)
            return 1
    bb = m()
    bb.name = '1a56d1a5'
    a = bb

    
    t=tic()
    print('\tusing joblib (40000000 length array)')
    print('\t\tsaving',end='')
    save_obj(a, 'weights\\2.model')
    toc(t)
    
    t=tic()
    print('\t\tloading',end='')
    model1 = load_obj('weights\\2.model')
    toc(t)
    print(model1.name)

    t=tic()
    print('\tusing pickle (40000000 length array)')
    
    print('\t\tsaving',end='')
    save_obj_pickle(a, 'weights\\1.model')
    toc(t)
    
    t=tic()
    print('\t\tloading',end='')
    b = load_obj_pickle('weights\\1.model')
    toc(t)


    '''
        using joblib (40000000 length array)
                saving  ----[tic-toc] costs 0.12466621398925781 s
                loading ----[tic-toc] costs 0.10471987724304199 s
        using pickle (40000000 length array)
                saving  ----[tic-toc] costs 2.895397424697876 s
                loading ----[tic-toc] costs 0.1216726303100586 s
    '''
    
    