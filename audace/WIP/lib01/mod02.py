#from ..jupytools import iprint

print("IMPORT MOD02")
print('__file__={0:<35} | __name__={1:<25} | __package__={2:<25}'.format(__file__,__name__,str(__package__)))

from codelib.lib02.mod03 import f3

def f2():
    print('In function f2 in mod02')
    f3()
    return