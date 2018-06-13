# An experiment to write a few scanpy (https://github.com/theislab/scanpy) APIs in Spark form

import numpy as np

from anndata import AnnData
from anndata_spark import *
from functools import partial
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
    if isinstance(data, AnnData):
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


def filter_cells(data, min_counts=None, min_genes=None, max_counts=None,
                 max_genes=None, copy=False):
    """Filter cell outliers based on counts and numbers of genes expressed.

    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers, i.e.,
    "unreliable" observations.

    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.

    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, `np.ndarray`, `sp.spmatrix`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts : `int`, optional (default: `None`)
        Minimum number of counts required for a cell to pass filtering.
    min_genes : `int`, optional (default: `None`)
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts : `int`, optional (default: `None`)
        Maximum number of counts required for a cell to pass filtering.
    max_genes : `int`, optional (default: `None`)
        Maximum number of genes expressed required for a cell to pass filtering.
    copy : `bool`, optional (default: `False`)
        If an :class:`~scanpy.api.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    If `data` is an :class:`~scanpy.api.AnnData`, filters the object and adds\
    either `n_genes` or `n_counts` to `adata.obs`. Otherwise a tuple

    cell_subset : `np.ndarray`
        Boolean index mask that does filtering. `True` means that the cell is
        kept. `False` means the cell is removed.
    number_per_cell : `np.ndarray`
        Either `n_counts` or `n_genes` per cell.

    Examples
    --------
    >>> adata = sc.datasets.krumsiek11()
    >>> adata.n_obs
    640
    >>> adata.var_names
    ['Gata2' 'Gata1' 'Fog1' 'EKLF' 'Fli1' 'SCL' 'Cebpa'
     'Pu.1' 'cJun' 'EgrNab' 'Gfi1']
    >>> # add some true zeros
    >>> adata.X[adata.X < 0.3] = 0
    >>> # simply compute the number of genes per cell
    >>> sc.pp.filter_cells(adata, min_genes=0)
    >>> adata.n_obs
    640
    >>> adata.obs['n_genes'].min()
    1
    >>> # filter manually
    >>> adata_copy = adata[adata.obs['n_genes'] >= 3]
    >>> adata_copy.obs['n_genes'].min()
    >>> adata.n_obs
    554
    >>> adata.obs['n_genes'].min()
    3
    >>> # actually do some filtering
    >>> sc.pp.filter_cells(adata, min_genes=3)
    >>> adata.n_obs
    554
    >>> adata.obs['n_genes'].min()
    3
    """
    if min_genes is not None and min_counts is not None:
        raise ValueError('Either provide min_counts or min_genes, but not both.')
    if min_genes is not None and max_genes is not None:
        raise ValueError('Either provide min_genes or max_genes, but not both.')
    if min_counts is not None and max_counts is not None:
        raise ValueError('Either provide min_counts or max_counts, but not both.')
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        cell_subset, number = filter_cells(adata.X, min_counts, min_genes, max_counts, max_genes)
        if min_genes is None and max_genes is None: adata.obs['n_counts'] = number
        else: adata.obs['n_genes'] = number
        adata._inplace_subset_obs(cell_subset)
        return adata if copy else None
    #
    # special case for Spark
    #
    elif isinstance(data, AnnDataRdd):
        adata = data.copy() if copy else data
        filter_cells_partial = partial(filter_cells_spark, min_counts=min_counts, min_genes=min_genes, max_counts=max_counts, max_genes=max_genes)
        result_rdd = adata.rdd.map(filter_cells_partial) # distributed computation
        result_rdd.cache()
        result = result_rdd.map(lambda t: (t[0], t[1])).collect() # retrieve per-partition cell_subset and numbers
        cell_subset = np.concatenate([res[0] for res in result])
        number = np.concatenate([res[1] for res in result])
        if min_genes is None and max_genes is None: adata.adata.obs['n_counts'] = number
        else: adata.adata.obs['n_genes'] = number
        adata.adata._inplace_subset_obs(cell_subset)
        adata.rdd = result_rdd.map(lambda t: t[2]) # compute filtered RDD
        return adata if copy else None
    #
    # end special case for Spark
    #
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = np.sum(X if min_genes is None and max_genes is None
                             else X > 0, axis=1)
    if issparse(X): number_per_cell = number_per_cell.A1
    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    s = np.sum(~cell_subset)
    print('filtered out {} cells that have'.format(s))
    if min_genes is not None or min_counts is not None:
        print('less than',
                 str(min_genes) + ' genes expressed'
                 if min_counts is None else str(min_counts) + ' counts')
    if max_genes is not None or max_counts is not None:
        print('more than ',
                 str(max_genes) + ' genes expressed'
                 if max_counts is None else str(max_counts) + ' counts')
    return cell_subset, number_per_cell

def filter_cells_spark(data, min_counts=None, min_genes=None, max_counts=None,
                 max_genes=None, copy=False):
    # differs from non-Spark version in that it returns the subsetted version of X too
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = np.sum(X if min_genes is None and max_genes is None
                             else X > 0, axis=1)
    if issparse(X): number_per_cell = number_per_cell.A1
    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    s = np.sum(~cell_subset)
    print('filtered out {} cells that have'.format(s))
    if min_genes is not None or min_counts is not None:
        print('less than',
              str(min_genes) + ' genes expressed'
              if min_counts is None else str(min_counts) + ' counts')
    if max_genes is not None or max_counts is not None:
        print('more than ',
              str(max_genes) + ' genes expressed'
              if max_counts is None else str(max_counts) + ' counts')
    return cell_subset, number_per_cell, X[cell_subset, :]