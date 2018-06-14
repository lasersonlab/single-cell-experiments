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
        filter_cells_partial = partial(_filter_cells_spark, min_counts=min_counts, min_genes=min_genes, max_counts=max_counts, max_genes=max_genes)
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

def _filter_cells_spark(data, min_counts=None, min_genes=None, max_counts=None,
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


def filter_genes(data, min_counts=None, min_cells=None, max_counts=None,
                 max_cells=None, copy=False):
    """Filter genes based on number of cells or counts.

    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.

    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.

    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, `np.ndarray`, `sp.spmatrix`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts : `int`, optional (default: `None`)
        Minimum number of counts required for a cell to pass filtering.
    min_cells : `int`, optional (default: `None`)
        Minimum number of cells expressed required for a cell to pass filtering.
    max_counts : `int`, optional (default: `None`)
        Maximum number of counts required for a cell to pass filtering.
    max_cells : `int`, optional (default: `None`)
        Maximum number of cells expressed required for a cell to pass filtering.
    copy : `bool`, optional (default: `False`)
        If an :class:`~scanpy.api.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    If `data` is an :class:`~scanpy.api.AnnData`, filters the object and adds\
    either `n_cells` or `n_counts` to `adata.var`. Otherwise a tuple

    gene_subset : `np.ndarray`
        Boolean index mask that does filtering. `True` means that the gene is
        kept. `False` means the gene is removed.
    number_per_cell : `np.ndarray`
        Either `n_counts` or `n_cells` per cell.
    """
    n_given_options = sum(
        option is not None for option in
        [min_cells, min_counts, max_cells, max_counts])
    if n_given_options != 1:
        raise ValueError(
            'Only provide one of the optional parameters `min_counts`,'
            '`min_cells`, `max_counts`, `max_cells` per call.')

    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        gene_subset, number = filter_genes(adata.X, min_cells=min_cells,
                                           min_counts=min_counts, max_cells=max_cells,
                                           max_counts=max_counts)
        if min_cells is None and max_cells is None:
            adata.var['n_counts'] = number
        else:
            adata.var['n_cells'] = number
        adata._inplace_subset_var(gene_subset)
        return adata if copy else None
    #
    # special case for Spark
    #
    elif isinstance(data, AnnDataRdd):
        adata = data.copy() if copy else data
        filter_genes_partial = partial(filter_genes, min_counts=min_counts, min_cells=min_cells, max_counts=max_counts, max_cells=max_cells)
        # First pass - count numbers
        result_rdd = adata.rdd.map(filter_genes_partial) # distributed computation
        result = result_rdd.collect() # retrieve per-partition numbers (ignore gene_subset and calculate on the driver)
        number = np.sum([res[1] for res in result], axis=0)
        min_number = min_counts if min_cells is None else min_cells
        max_number = max_counts if max_cells is None else max_cells
        if min_number is not None:
            gene_subset = number >= min_number
        if max_number is not None:
            gene_subset = number <= max_number
        if min_cells is None and max_cells is None:
            adata.adata.var['n_counts'] = number
        else:
            adata.adata.var['n_cells'] = number
        adata.adata._inplace_subset_var(gene_subset)
        # Second pass - filter columns by gene_subset
        adata.rdd = adata.rdd.map(_apply_gene_subset(gene_subset)) # compute filtered RDD
        return adata if copy else None
    #
    # end special case for Spark
    #
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_cells is None else min_cells
    max_number = max_counts if max_cells is None else max_cells
    number_per_gene = np.sum(X if min_cells is None and max_cells is None
                             else X > 0, axis=0)
    if issparse(X):
        number_per_gene = number_per_gene.A1
    if min_number is not None:
        gene_subset = number_per_gene >= min_number
    if max_number is not None:
        gene_subset = number_per_gene <= max_number

    s = np.sum(~gene_subset)
    print('filtered out {} genes that are detected'.format(s))
    if min_cells is not None or min_counts is not None:
        print('in less than',
                 str(min_cells) + ' cells'
                 if min_counts is None else str(min_counts) + ' counts')
    if max_cells is not None or max_counts is not None:
        print('in more than ',
                 str(max_cells) + ' cells'
                 if max_counts is None else str(max_counts) + ' counts')
    return gene_subset, number_per_gene

def _apply_gene_subset(gene_subset):
    def subset(data):
        X = data
        return X[:, gene_subset]
    return subset

def scale(data, zero_center=True, max_value=None, copy=False):
    """Scale data to unit variance and zero mean.

    Parameters
    ----------
    data : :class:`~scanpy.api.AnnData`, `np.ndarray`, `sp.sparse`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    zero_center : `bool`, optional (default: `True`)
        If `False`, omit zero-centering variables, which allows to handle sparse
        input efficiently.
    max_value : `float` or `None`, optional (default: `None`)
        Clip (truncate) to this value after scaling. If `None`, do not clip.
    copy : `bool`, optional (default: `False`)
        If an :class:`~scanpy.api.AnnData` is passed, determines whether a copy
        is returned.

    Returns
    -------
    Depending on `copy` returns or updates `adata` with a scaled `adata.X`.
    """
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        # need to add the following here to make inplace logic work
        if zero_center and issparse(adata.X):
            print(
                '... scale_data: as `zero_center=True`, sparse input is '
                'densified and may lead to large memory consumption')
            adata.X = adata.X.toarray()
        scale(adata.X, zero_center=zero_center, max_value=max_value, copy=False)
        return adata if copy else None
    #
    # special case for Spark
    #
    elif isinstance(data, AnnDataRdd):
        adata = data.copy() if copy else data
        adata.rdd = _scale_spark(adata.rdd, zero_center=zero_center) # TODO: support max_value
        return adata if copy else None
    #
    # end special case for Spark
    #
    X = data.copy() if copy else data  # proceed with the data matrix
    zero_center = zero_center if zero_center is not None else False if issparse(X) else True
    if not zero_center and max_value is not None:
        print(
            '... scale_data: be careful when using `max_value` without `zero_center`')
    if max_value is not None:
        print('... clipping at max_value', max_value)
    if zero_center and issparse(X):
        print('... scale_data: as `zero_center=True`, sparse input is '
                 'densified and may lead to large memory consumption, returning copy')
        X = X.toarray()
        copy = True
    _scale(X, zero_center)
    if max_value is not None: X[X > max_value] = max_value
    return X if copy else None

def _get_mean_var(X):
    # - using sklearn.StandardScaler throws an error related to
    #   int to long trafo for very large matrices
    # - using X.multiply is slower
    if True:
        mean = X.mean(axis=0)
        if issparse(X):
            mean_sq = X.multiply(X).mean(axis=0)
            mean = mean.A1
            mean_sq = mean_sq.A1
        else:
            mean_sq = np.multiply(X, X).mean(axis=0)
        # enforece R convention (unbiased estimator) for variance
        var = (mean_sq - mean**2) * (X.shape[0]/(X.shape[0]-1))
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False).partial_fit(X)
        mean = scaler.mean_
        # enforce R convention (unbiased estimator)
        var = scaler.var_ * (X.shape[0]/(X.shape[0]-1))
    return mean, var

def _get_mean_var_spark(rddX):
    result = rddX.map(_get_count_and_sums).collect()
    total_count = sum([res[0] for res in result])
    mean = np.sum([res[1] for res in result], axis=0) / total_count
    mean_sq = np.sum([res[2] for res in result], axis=0) / total_count
    var = (mean_sq - mean**2) * (total_count/(total_count-1))
    return mean, var

def _get_count_and_sums(X):
    # calculate count, sum, sum squared for columns in each chunk
    count = X.shape[0]
    sum = np.sum(X, axis=0)
    sum_sq = np.multiply(X, X).sum(axis=0)
    return count, sum, sum_sq

def _scale(X, zero_center=True):
    # - using sklearn.StandardScaler throws an error related to
    #   int to long trafo for very large matrices
    # - using X.multiply is slower
    #   the result differs very slightly, why?
    if True:
        mean, var = _get_mean_var(X)
        scale = np.sqrt(var)
        if issparse(X):
            if zero_center: raise ValueError('Cannot zero-center sparse matrix.')
            sparsefuncs.inplace_column_scale(X, 1/scale)
        else:
            X -= mean
            X /= scale
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=zero_center, copy=False).partial_fit(X)
        # user R convention (unbiased estimator)
        scaler.scale_ *= np.sqrt(X.shape[0]/(X.shape[0]-1))
        scaler.transform(X)

def _scale_spark(rddX, zero_center=True):
    mean, var = _get_mean_var_spark(rddX)
    scale = np.sqrt(var)
    return rddX.map(_scale_map_fn(mean, scale))

def _scale_map_fn(mean, scale):
    def scale_int(X):
        X -= mean
        X /= scale
        return X
    return scale_int