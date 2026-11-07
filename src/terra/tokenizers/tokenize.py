import numpy as np


def process_gene_tokens(gene_tokens: list[int],
                        length: int,
                        token_dict: dict,
                        ) -> tuple[np.ndarray, int]:
    """
    Add pad tokens or truncate gene token list based on length.

    Parameters
    ----------
    gene_tokens:
       List containing (ranked) gene tokens.
    length:
        Length to which to pad or truncate the gene token list to.
    token_dict:
        Token dictionary.

    Returns
    ----------
    processed_gene_tokens:
       Array containing padded or truncated (ranked) gene tokens.
    num_nonzero_tokens:
       Number of nonzero (non-pad) gene tokens.
    """
    processed_gene_tokens = np.array(gene_tokens, dtype=np.int64)
    
    pad_size = int(length - len(processed_gene_tokens))
    if pad_size < 0:
        # Truncate
        processed_gene_tokens = processed_gene_tokens[:length]
    else:
        # Add pad tokens
        processed_gene_tokens = np.pad(
            processed_gene_tokens,
            (0, pad_size),
            'constant',
            constant_values=token_dict.get('<pad>'))

    # Count the gene tokens rather than the list length: the dense ranking
    # path stores the full panel with non-expressed genes set to token 0, so
    # the list length overstates the number of genes (the length-grouped
    # sampler then groups cells by panel size instead of expressed genes).
    # Identical to the list length for the sparse path.
    num_nonzero_tokens = int(np.count_nonzero(
        (processed_gene_tokens != 0)
        & (processed_gene_tokens != token_dict.get('<pad>', 0))))
                
    return processed_gene_tokens, num_nonzero_tokens


def process_gene_expr(gene_expr: list[float],
                      length: int,
                      ) -> np.ndarray:
    """
    Pad gene expression with '0.' or truncate gene expression based on
    length.

    Parameters
    ----------
    gene_expr:
        List containing (ranked) gene expression.
    length:
        Length to which to pad or truncate the gene expression list to.

    Returns
    ----------
    processed_gene_expr:
        Array containing padded or truncated (ranked) gene expression.
    """
    processed_gene_expr = np.array(gene_expr)
    
    pad_size = int(length - len(processed_gene_expr))
    if pad_size < 0:
        # Truncate
        processed_gene_expr = processed_gene_expr[:length]
    else:
        # Add padding with 0s
        processed_gene_expr = np.pad(
            processed_gene_expr,
            (0, pad_size),
            'constant',
            constant_values=0.)
                
    return processed_gene_expr
    

def rank_gene_order(gene_scores: np.ndarray,
                    gene_tokens: np.ndarray,
                    stable: bool = False,
                    tie_break_values: np.ndarray | None = None,
                    ) -> np.ndarray:
    """
    Return the indices that rank genes by descending gene score.

    Parameters
    ----------
    gene_scores:
        1D vector containing gene scores (normalized gene expression).
    gene_tokens:
        1D vector containing the gene tokens matching `gene_scores`.
    stable:
        If `True`, order genes with tied scores by ascending token id, so the
        ranking does not depend on the order of the genes in the input.
        Default `False` uses numpy's default sort, whose order of tied genes
        is unspecified (it depends on the input order and the numpy build).
    tie_break_values:
        Optional 1D vector of per-gene values matching `gene_scores` (e.g. the
        corpus mean of each gene). With `stable`, genes with tied scores are
        ordered by ascending value first and by ascending token id after
        that. Requires `stable=True`.

    Returns
    ----------
    sorted_indices:
        1D vector of indices into `gene_scores` from the highest to the
        lowest score.
    """
    if tie_break_values is not None and not stable:
        raise ValueError("'tie_break_values' requires 'stable=True'.")
    if stable:
        if tie_break_values is not None:
            return np.lexsort((gene_tokens, tie_break_values, -gene_scores))
        return np.lexsort((gene_tokens, -gene_scores))
    return np.argsort(-gene_scores)


def rank_gene_tokens(gene_scores: np.ndarray,
                     gene_tokens: np.ndarray,
                     n_tokens: int | None = None,
                     stable: bool = False,
                     tie_break_values: np.ndarray | None = None,
                     ) -> np.ndarray:
    """
    Rank gene tokens based on matching gene scores (highest score ->
    rank 1).

    Parameters
    ----------
    gene_scores:
        1D vector containing gene scores (normalized gene expression).
    gene_tokens:
        1D vector containing gene tokens.
    n_tokens:
        Number of tokens to be returned.
    stable:
        If `True`, order genes with tied scores by ascending token id (see
        `rank_gene_order`). Default `False` uses numpy's default sort, whose
        order of tied genes is unspecified.
    tie_break_values:
        Optional per-gene values that order tied genes before the token id
        (see `rank_gene_order`). Requires `stable=True`.

    Returns
    ----------
    ranked_gene_tokens:
        1D vector containing gene tokens ranked by gene scores.
    """
    # Sort gene tokens by gene scores
    sorted_indices = rank_gene_order(gene_scores, gene_tokens, stable=stable,
                                     tie_break_values=tie_break_values)
    ranked_gene_tokens = gene_tokens[sorted_indices][:n_tokens]

    return ranked_gene_tokens
