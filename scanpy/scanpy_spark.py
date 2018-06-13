# An experiment to write a few scanpy (https://github.com/theislab/scanpy) APIs in Spark form

import anndata as ad
import numpy as np

from anndata_spark import *

from scipy.sparse import issparse

# c.f. http://scanpy.readthedocs.io/en/latest/api/scanpy.api.pp.log1p.html#scanpy.api.pp.log1p
def log1p(data, copy=False, chunked=False, chunk_size=None):
    """Logarithmize the data matrix.

    Computes `X = log(X + 1)`, where `log` denotes the natural logrithm.

    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, `np.ndarray`, `sp.sparse`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    copy : `bool`, optional (default: `False`)
        If an :class:`~scanpy.api.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    Returns or updates `data`, depending on `copy`.
    """
    if isinstance(data, ad.AnnData):
        adata = data.copy() if copy else data
        if chunked:
            for chunk, start, end in adata.chunked_X(chunk_size):
                adata.X[start:end] = log1p(chunk)
        else:
            adata.X = log1p(data.X)
        return adata if copy else None
    #
    # special case for Spark
    #
    elif isinstance(data, AnnDataRdd):
        adata = data.copy() if copy else data
        adata.rdd = adata.rdd.map(log1p) # recursive call to operate on numpy array
        return adata if copy else None
    #
    # end special case for Spark
    #
    X = data  # proceed with data matrix
    if not issparse(X):
        return np.log1p(X)
    else:
        return X.log1p()
