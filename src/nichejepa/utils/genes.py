import json
import requests


def get_ensembl_ids(gene_names: list,
                    species: str="homo_sapiens") -> dict:
    """
    Get gene Ensembl IDs based on gene names via Ensembl REST API.

    Parameters
    ----------
    gene_names:
        List of gene names.
    species:
        Species, e.g. homo_sapiens or mus_musculus.

    Returns
    ----------
    ensembl_ids:
        Dictionary where keys are gene names and values are Ensembl IDs.
    """
    server = "https://rest.ensembl.org"
    endpoint = f"/lookup/symbol/{species}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    data = {"symbols": gene_names}
    response = requests.post(f"{server}{endpoint}", headers=headers, data=json.dumps(data))
    
    if response.ok:
        ensembl_ids = {}
        for key, value in response.json().items():
            ensembl_ids[key] = value["id"]
        if len(ensembl_ids.keys()) != len(gene_names):
            missing_genes = [gene for gene in gene_names if gene not in ensembl_ids.keys()]
            print(f"Could not find Ensembl IDs for genes: {missing_genes}.")
        return ensembl_ids
    else:
        response.raise_for_status()
