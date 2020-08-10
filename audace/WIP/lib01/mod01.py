from ..jupytools import iprint
print("IMPORT MOD01")
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))

from .mod02 import f2


def f1():
    print('In function f1 in mod01')
    f2()
    return