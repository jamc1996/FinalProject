""" Script for the CgSvm class which provide python interface with C functions.
"""

from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer 

from ctypes import (
    CDLL,
    POINTER,
    addressof,
    c_char_p,
    c_double,
    c_float,
    c_int,
)
from structs import (
    Projected,
    Fullproblem,
    svmModel,
)


_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C') 

def numpy_to_c_types(x):
    return x.ctypes.data_as(POINTER(c_double))

def c_types_to_numpy(array, array_length):
    return np.ctypeslib.as_array(
        (
            c_double * array_length
        ).from_address(
            addressof(array.contents)
        )
    )

_CGSVM_FUNCTIONS = [
    {
        'funcname': 'calcYTR',
        'restype': None,
        'argtypes': [POINTER(Projected), POINTER(Fullproblem)],
    },
    {
        'funcname': 'copyVector',
        'restype': None,
        'argtypes': [POINTER(c_double), POINTER(c_double), c_int],
    },
    {
        'funcname': 'fit',
        'restype': svmModel,
        'argtypes': [
            _doublepp,
            c_int,
            c_int,
            c_int,
            c_int,
            c_char_p,
            POINTER(c_double),
        ],
    },
]

_CGSVM_TRANSFORMER_FUNCTIONS = [
    {
        'funcname': 'transform',
        'restype': None,
        'argtypes': [_doublepp, c_int, svmModel, POINTER(c_double)],
    },
]

class CgSvmTransformer:
    def __init__(self, trained_model=None, decision_vector=None, saved_model_file=None):
        """ Set up the class with function interfaces.
        """
        if trained_model is None and saved_model_file is None:
            raise ValueError('A model must be passed.')

        if trained_model is not None:
            self.trained_model = trained_model
            self.decision_vector = decision_vector

        libname = Path().absolute().parent / "Serial/pygert"
        self.c_lib = CDLL(libname)
        for cgsvm_func in _CGSVM_TRANSFORMER_FUNCTIONS:
            setattr(
                self,
                f'_{cgsvm_func["funcname"]}',
                self._wrap_function(
                    funcname=cgsvm_func['funcname'],
                    restype=cgsvm_func['restype'],
                    argtypes=cgsvm_func['argtypes']
                )
            )

    def transform(
            self,
            arranged_pandas_df
        ):
        arranged_numpy_array = arranged_pandas_df.to_numpy(dtype=np.float64, na_value=0.0)
        array_pp = (
            arranged_numpy_array.__array_interface__['data'][0] + 
            np.arange(arranged_numpy_array.shape[0])*arranged_numpy_array.strides[0]
        ).astype(np.uintp)
        array_length = arranged_numpy_array.shape[0]
        output_vector = np.zeros(array_length, dtype=np.float64)
        c_output_vector = numpy_to_c_types(output_vector)

        self._transform(
            array_pp,
            array_length,
            self.trained_model,
            c_output_vector,
        )
        return c_types_to_numpy(c_output_vector, array_length)

    def _wrap_function(self, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = self.c_lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func


class CgSvm:
    def __init__(self):
        """ Set up the class with function interfaces.
        """
        libname = Path().absolute().parent / "Serial/pygert"
        self.c_lib = CDLL(libname)
        # 
        for cgsvm_func in _CGSVM_FUNCTIONS:
            setattr(
                self,
                f'_{cgsvm_func["funcname"]}',
                self._wrap_function(
                    funcname=cgsvm_func['funcname'],
                    restype=cgsvm_func['restype'],
                    argtypes=cgsvm_func['argtypes']
                )
            )
    
    def copyVector(self, array_to_copy, template_array, array_length):
        ctype_array_to_copy = numpy_to_c_types(array_to_copy)
        ctype_template_array = numpy_to_c_types(template_array)
        self._copy_vector(ctype_array_to_copy, ctype_template_array, array_length)
        copied_array = c_types_to_numpy(ctype_array_to_copy, array_length)
        template_array = c_types_to_numpy(ctype_template_array, array_length)
        return copied_array, template_array

    def fit(
            self,
            arranged_pandas_df,
            n_positive,
            save_to_file=False,
            file_name=''
        ):
        if save_to_file and not file_name:
            raise ValueError('If saving to file, file_name must be set.')

        arranged_numpy_array = arranged_pandas_df.to_numpy(dtype=np.float64, na_value=0.0)
        n_instances = arranged_numpy_array.shape[0]
        n_features = arranged_numpy_array.shape[1]

        array_pp = (
            arranged_numpy_array.__array_interface__['data'][0] + 
            np.arange(arranged_numpy_array.shape[0])*arranged_numpy_array.strides[0]
        ).astype(np.uintp)

        decision_vector = np.empty(n_instances, dtype=np.float64)
        ctype_dv_array = numpy_to_c_types(decision_vector)

        trained_model = self._fit(
            array_pp,
            n_features,
            n_instances,
            n_positive,
            int(save_to_file),
            file_name.encode('utf-8'),
            ctype_dv_array
        )
        return CgSvmTransformer(
            trained_model=trained_model,
            decision_vector=decision_vector
        )

    def _wrap_function(self, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = self.c_lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func


