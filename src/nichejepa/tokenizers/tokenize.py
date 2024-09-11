from typing import Optional

import numpy as np


def process_gene_tokens(gene_tokens: list,
                        length: int,
                        token_dict: dict,
                        special_tokens: Optional[list],
                        special_tokens_idx: Optional[list]
                        ) -> list:
    """
    Add pad tokens or truncate gene token list based on length and add special
    tokens if defined.

    Parameters
    ----------
    gene_tokens:
       List containing (ranked) gene tokens.
    length:
        Length to which to pad or truncate the gene token list to.
    token_dict:
        Token dictionary.
    special_tokens:
        List of special tokens to be added to the gene token list.
    special_tokens_idx:
        List with indices where special tokens are added to the gene token list.

    Returns
    ----------
    processed_gene_tokens:
       List containing padded or truncated (ranked) gene tokens, including
       special tokens if defined.       
    """

    if special_tokens:
        # Make space for special tokens
        processed_gene_tokens = gene_tokens[:(length-len(special_tokens))]

        # Add special tokens
        for special_token, special_token_idx in zip(
            special_tokens, special_tokens_idx):
            processed_gene_tokens = np.insert(
                processed_gene_tokens,
                special_token_idx,
                token_dict.get(special_token))
    else:
        processed_gene_tokens = gene_tokens

    # Convert to np.int64 to ensure all elements are of the same type. Should
    # this be double?
    processed_gene_tokens = np.array(processed_gene_tokens, dtype=np.int64)
    
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
            constant_values=token_dict.get("<pad>"))
        
    return processed_gene_tokens
    

def rank_gene_tokens(gene_scores: np.ndarray,
                     gene_tokens: np.ndarray,
                     n_tokens: Optional[int]=None
                     ) -> np.ndarray:
    """
    Rank gene tokens based on matching gene scores (highest gene score -> rank 1
    gene).

    Parameters
    ----------
    gene_scores:
        1D vector containing gene scores (read depth normalized gene expression
        scaled by means and regularizing standard deviations).
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