"""The 96 reachable nonterminal quotient cells defined by v8.3.1.

Non-search navigation is identically one. Known successful terminal value is
one and therefore needs no empirical cell. The estimator is in value_table.py.
"""
from .tracker import STAGES,Task,is_search


def _search(family,p):
    return p==0 or (family=='pick_two_obj_and_place' and p==2) or (family=='look_at_obj_in_light' and p==1)


CELLS=tuple((family,p,nav,budget)
            for family,K in STAGES.items()
            for p in range(-1,K)
            for nav in ([0,1] if _search(family,p) else [1])
            for budget in range(3))
CELL_INDEX={cell:i for i,cell in enumerate(CELLS)}
assert len(CELLS)==len(CELL_INDEX)==96


def cell_index(family,p,nav,budget):
    if family not in STAGES:raise ValueError('Unknown ALFWorld family')
    if p==STAGES[family]:return None
    key=(family,p,nav,budget)
    if key not in CELL_INDEX:raise ValueError(f'Unreachable or invalid quotient cell: {key}')
    return CELL_INDEX[key]
