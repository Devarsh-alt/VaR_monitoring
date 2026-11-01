"""Single-source compatibility exports; no metric algorithm is rewritten."""
from .._frozen import load
def rank_metrics(*a,**kw):return load("evaluation").rank_metrics(*a,**kw)
def metrics(*a,**kw):return load("evaluation").metrics(*a,**kw)
def matched(*a,**kw):return load("evaluation").matched(*a,**kw)
