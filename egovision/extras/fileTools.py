import sys,os

def fullVideoName(name):
    pt, name = os.path.split(name)
    files = os.listdir(pt)
    for f in files:
        if name == f[:-4]:
            return os.path.abspath(pt + "/" + f)
    raise NameError("File not founded")
