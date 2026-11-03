import logging
import os

import anndata as ad
import pandas as pd


logger = logging.getLogger(__name__)

# The ligand-receptor utilities need the optional ``eval`` extra (cellphonedb,
# omnipath, ipython). They are imported where they are used, so that the rest
# of ``terra.evaluation`` (e.g. the linear and kNN probes) imports without the
# extra, and without the network requests omnipath makes when it is imported.
_EVAL_EXTRA = 'Install it with `pip install "terra-st[eval]"`.'


def _display_html(html: str) -> None:
    """Display HTML in a notebook (no-op without IPython)."""
    try:
        from IPython.display import HTML, display
    except ModuleNotFoundError:
        logger.info('IPython is not installed; not displaying the table.')
        return
    display(HTML(html))


def _split_ligand_receptor(interactors: str,
                           gene_names: set[str],
                           ) -> tuple[str | None, str | None]:
    """
    Split a CellPhoneDB ``'<ligand>-<receptor>'`` pair of gene symbols.

    Gene symbols can contain hyphens themselves (e.g. ``HLA-E`` in
    ``HLA-E-KLRC1``), so the pair is split at the hyphen that leaves one of
    ``gene_names`` on both sides. Returns ``(None, None)`` if no split (or
    more than one) does.
    """
    parts = interactors.split('-')
    matches = [
        ('-'.join(parts[:i]), '-'.join(parts[i:]))
        for i in range(1, len(parts))
        if ('-'.join(parts[:i]) in gene_names
            and '-'.join(parts[i:]) in gene_names)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            f"CellPhoneDB interactors '{interactors}' can be split into "
            f"several gene pairs of the AnnData object ({matches}); "
            "skipping it.")
    return None, None


def get_adata_cellphonedb_lr_pairs(
        adata: ad.AnnData,
        save_folder_path: str,
        cpdb_version: str = 'v5.0.0',
        legacy_split: bool = False,
        ) -> pd.DataFrame:
    """
    Retrieve ligand-receptor pairs from CellPhoneDB
    (https://www.cellphonedb.org/) that are present in an AnnData object.

    Requires the optional ``eval`` extra (``pip install "terra-st[eval]"``).

    Parameters
    -----------
    adata:
        The AnnData object to check for gene availability. ``adata.var`` is
        indexed by gene symbol and has an ``ensembl_id`` column (as returned
        by ``harmonize_adata``).
    save_folder_path:
        Folder into which the CellPhoneDB database is downloaded.
    cpdb_version:
        CellPhoneDB database version.
    legacy_split:
        If `True`, split the CellPhoneDB ``interactors`` at their first
        hyphen, as earlier versions did. This drops every pair with a
        hyphenated gene symbol (e.g. ``HLA-E-KLRC1`` became ``HLA`` and
        ``E``); use it only to reproduce earlier results. By default the
        pair is split at the hyphen that leaves a gene of ``adata`` on both
        sides.

    Returns
    -----------
    lr_interaction_df:
        A DataFrame containing ligand and receptor pairs present in the
        AnnData object (Ensembl IDs).
    """
    try:
        from cellphonedb.utils import db_releases_utils, db_utils
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"get_adata_cellphonedb_lr_pairs requires 'cellphonedb'. "
            f"{_EVAL_EXTRA}") from e

    # Display CellPhoneDB version
    _display_html(db_releases_utils.get_remote_database_versions_html()[
        'db_releases_html_table'])

    # Path where the input files to generate the database are/will be located
    os.makedirs(save_folder_path, exist_ok=True)
    cpdb_target_dir = os.path.join(save_folder_path, cpdb_version)

    # Download the CellPhoneDB database
    db_utils.download_database(cpdb_target_dir, cpdb_version)

    # Retrieve CellPhoneDB interactions
    intercell_df = pd.read_csv(f'{cpdb_target_dir}/interaction_input.csv')

    # Filter for lr interactions
    lr_interaction_df = intercell_df[
        intercell_df['directionality'] == 'Ligand-Receptor']

    # Remove interactions involving protein complexes
    lr_interaction_df = lr_interaction_df[
        ~lr_interaction_df['interactors'].apply(lambda x: '+' in x)]

    # Only keep ligand and receptor gene names
    if legacy_split:
        ligands = [x.split('-')[0] for x in lr_interaction_df['interactors']]
        receptors = [x.split('-')[1] for x in lr_interaction_df['interactors']]
    else:
        gene_names = set(adata.var.index)
        pairs = [_split_ligand_receptor(x, gene_names)
                 for x in lr_interaction_df['interactors']]
        ligands = [ligand for ligand, _ in pairs]
        receptors = [receptor for _, receptor in pairs]
    lr_interaction_df = pd.DataFrame(
        {'ligand': ligands, 'receptor': receptors},
        index=lr_interaction_df.index)

    # Map gene names to ensembl IDs, keeping only genes present in the AnnData
    # object
    lr_interaction_df['ligand'] = lr_interaction_df[
        'ligand'].map(adata.var['ensembl_id'])
    lr_interaction_df['receptor'] = lr_interaction_df[
        'receptor'].map(adata.var['ensembl_id'])

    # Drop interactions where ligand or receptor is not in the AnnData object
    lr_interaction_df = lr_interaction_df.dropna()

    return lr_interaction_df


def get_adata_omnipath_lr_pairs(
        adata: ad.AnnData,
        min_curation_effort: int = 2
        ) -> pd.DataFrame:
    """
    Retrieve ligand-receptor pairs from OmniPath (https://omnipathdb.org/) that
    are present in an AnnData object.

    Requires the optional ``eval`` extra (``pip install "terra-st[eval]"``).

    Parameters
    -----------
    adata:
        The AnnData object to check for gene availability
    min_curation_effort:
        Minimum curation effort parameter of the OmniPath database.

    Returns
    -----------
    lr_interaction_df:
        A DataFrame containing ligand and receptor pairs present in the
        AnnData object.
    """
    try:
        import omnipath as op
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"get_adata_omnipath_lr_pairs requires 'omnipath'. "
            f"{_EVAL_EXTRA}") from e

    # Retrieve OmniPath interactions
    intercell_df = op.interactions.import_intercell_network(
        include=['omnipath', 'pathwayextra', 'ligrecextra'])

    # Filter for lr interactions
    lr_interaction_df = intercell_df[
        (intercell_df['category_intercell_source'] == 'ligand') &
        (intercell_df['category_intercell_target'] == 'receptor')]

    # Filter for minimum curation effort
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df['curation_effort'] >= min_curation_effort]

    # Remove interactions involving protein complexes
    lr_interaction_df = lr_interaction_df[
        ~lr_interaction_df['genesymbol_intercell_source'].apply(
            lambda x: 'COMPLEX' in x)]
    lr_interaction_df = lr_interaction_df[
        ~lr_interaction_df['genesymbol_intercell_target'].apply(
            lambda x: 'COMPLEX' in x)]

    # Only keep ligand and receptor gene names
    lr_interaction_df = lr_interaction_df[
        ['genesymbol_intercell_source', 'genesymbol_intercell_target']].copy()

    # Map gene names to ensembl IDs, keeping only genes present in the AnnData
    # object
    lr_interaction_df['ligand'] = lr_interaction_df[
        'genesymbol_intercell_source'].map(adata.var['ensembl_id'])
    lr_interaction_df['receptor'] = lr_interaction_df[
        'genesymbol_intercell_target'].map(adata.var['ensembl_id'])
    lr_interaction_df.drop(
        ['genesymbol_intercell_source', 'genesymbol_intercell_target'],
        axis=1,
        inplace=True)

    # Drop interactions where ligand or receptor is not in the AnnData object
    lr_interaction_df = lr_interaction_df.dropna()

    return lr_interaction_df