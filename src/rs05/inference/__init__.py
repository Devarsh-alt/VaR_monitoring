from .._frozen import load
def weights(*a,**kw):return load("inference").weights(*a,**kw)
def distribution(*a,**kw):return load("inference").distribution(*a,**kw)
def summary(*a,**kw):return load("inference").summary(*a,**kw)
def holm(*a,**kw):return load("inference").holm(*a,**kw)
