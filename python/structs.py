""" Suite of python classes (inheriting from Structure) which map 
to the structs used in the cgsvm C library.
"""
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_int,
)

class Cell_struct(Structure):
    pass
Cell_struct._fields_ = [
    ('next', POINTER(Cell_struct)),
    ('prev', POINTER(Cell_struct)),
    ('label', c_int),
    ('line', POINTER(c_double)),
]

class denseData(Structure):
    _fields_ = [
        ('nInstances', c_int),
        ('nFeatures', c_int),
        ('nPos', c_int),
        ('nNeg', c_int),
        ('data', POINTER(POINTER(c_double))),
        ('data1d', POINTER(Cell_struct)),
    ]

class List(Structure):
    _fields_ = [
        ('head', POINTER(Cell_struct)),
        ('tail', POINTER(Cell_struct)),
    ]

class svmModel(Structure):
    _fields_ = [
        ('kernel', c_int),
        ('trainElapsedTime', c_int),
        ('nFeatures', c_int),
        ('decisionVector', POINTER(c_double)),
        ('biasTerm', c_double),
    ]

class Fullproblem(Structure):
    _fields_ = [
        ('n', c_int),
        ('p', c_int),
        ('q', c_int),
        ('C', c_double),

        ('alpha', POINTER(c_double)),
        ('beta', POINTER(c_double)),
        ('gradF', POINTER(c_double)),
        ('active', POINTER(c_int)),
        ('inactive', POINTER(c_int)),

        ('partialH', List),
    ]

class Projected(Structure):
    _fields_ = [
        ('alphaHat', POINTER(c_double)),
        ('yHat', POINTER(c_double)),
        ('rHat', POINTER(c_double)),
        ('H', POINTER(c_double)),
        ('gamma', POINTER(c_double)),
        ('rho', POINTER(c_double)),
        ('Hrho', POINTER(c_double)),
        ('p', c_int),
        ('C', c_double),
        ('h', POINTER(c_double)),
        ('ytr', c_double),
    ]
