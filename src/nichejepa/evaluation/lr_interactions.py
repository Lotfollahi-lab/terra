import os

import anndata as ad
import omnipath as op
import pandas as pd
from cellphonedb.utils import db_releases_utils, db_utils
from IPython.display import HTML, display


def get_adata_cellphonedb_lr_pairs(
        adata: ad.AnnData,
        save_folder_path: str,
        cpdb_version: str = 'v5.0.0',
        ) -> pd.DataFrame:
    """
    Retrieve ligand-receptor pairs from CellPhoneDB
    (https://www.cellphonedb.org/) that are present in an AnnData object.

    Parameters
    -----------
    adata:
        The AnnData object to check for gene availability
    save_folder_path

    Returns
    -----------
    lr_interaction_df:
        A DataFrame containing ligand and receptor pairs present in the
        AnnData object.
    """
    # Display CellPhoneDB version
    display(HTML(db_releases_utils.get_remote_database_versions_html()[
        'db_releases_html_table']))

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
    lr_interaction_df['ligand'] = lr_interaction_df[
        'interactors'].apply(lambda x: x.split('-')[0])
    lr_interaction_df['receptor'] = lr_interaction_df[
        'interactors'].apply(lambda x: x.split('-')[1])
    lr_interaction_df = lr_interaction_df[['ligand', 'receptor']]

    # Map gene names to ensembl IDs, keeping only genes present in the AnnData
    # object
    lr_interaction_df['ligand'] = lr_interaction_df[
        'ligand'].map(adata.var['ensembl_id'])
    lr_interaction_df['receptor'] = lr_interaction_df[
        'receptor'].map(adata.var['ensembl_id'])

    print()

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
        ['genesymbol_intercell_source', 'genesymbol_intercell_target']]

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