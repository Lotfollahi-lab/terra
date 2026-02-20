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
       Number of nonzero gene tokens.
    """
    processed_gene_tokens = np.array(gene_tokens, dtype=np.int64)
    
    pad_size = int(length - len(processed_gene_tokens))
    if pad_size < 0:
        # Truncate
        processed_gene_tokens = processed_gene_tokens[:length]
        num_nonzero_tokens = length
    else:
        # Add pad tokens
        processed_gene_tokens = np.pad(
            processed_gene_tokens,
            (0, pad_size),
            'constant',
            constant_values=token_dict.get('<pad>'))
        num_nonzero_tokens = len(processed_gene_tokens) - pad_size
                
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
    

def rank_gene_tokens(gene_scores: np.ndarray,
                     gene_tokens: np.ndarray,
                     n_tokens: int | None = None,
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

    Returns
    ----------
    ranked_gene_tokens:
        1D vector containing gene tokens ranked by gene scores.
    """
    # Sort gene tokens by gene scores
    sorted_indices = np.argsort(-gene_scores)
    ranked_gene_tokens = gene_tokens[sorted_indices][:n_tokens]
    
    return ranked_gene_tokens
