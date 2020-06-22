import numpy as np
from numpy import loadtxt

DATA = '500_Person_Gender_Height_Weight_Index.csv'


def get_data_h():
    g, h, w, i = loadtxt(
        DATA,
        dtype={
            'names': ['Gender', 'Height', 'Weight', 'Index'],
            'formats': ['S1', 'i4', 'i4', 'i4']
        },
        unpack=True,
        skiprows=1,
        delimiter=','
    )

    c = lambda x: 0 if x < 160 else 1 if x < 180 else 2
    h = [c(hs) for hs in h]

    d = np.array([g, h]).T
    return d


def get_data():
    g, h, w, i = loadtxt(
        DATA,
        dtype={
            'names': ['Gender', 'Height', 'Weight', 'Index'],
            'formats': ['S1', 'i4', 'i4', 'i4']
        },
        unpack=True,
        skiprows=1,
        delimiter=','
    )

    hg = lambda x: 0 if x == 'F' else 1
    g = [hg(hs) for hs in g]

    hc = lambda x: 2 if x < 160 else 3 if x < 180 else 4
    h = [hc(hs) for hs in h]

    wc = lambda x: 5 if x < 45 else 6 if x < 60 else 7 if x < 80 else 8 if x < 100 else 9
    w = [wc(ws) for ws in w]

    d = np.array([g, h, w]).T
    return d
