"""push_model_to_hub with a recording stand-in for huggingface_hub.HfApi (no
network): the model card is never overwritten by a README.md of the bundle
folder, and an existing tag is neither moved nor silently kept."""
import fnmatch
import types

import pytest

import terra.hub as hub

BUNDLE = ['model_checkpoint.pt', 'model_config.yaml', 'token_dictionary.pkl',
          'ensembl_dictionary.pkl', 'gene_count_dictionary.pkl']


class FakeApi:
    """Records the calls; the repo has the given tags."""

    def __init__(self, tags=None, exists=True):
        self.tags = dict(tags or {})
        self.exists = exists
        self.calls = []

    def list_repo_refs(self, repo_id, repo_type=None):
        if not self.exists:
            from huggingface_hub.errors import RepositoryNotFoundError
            # (its constructor needs an HTTP response)
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        return types.SimpleNamespace(tags=[
            types.SimpleNamespace(name=name, target_commit=commit)
            for name, commit in self.tags.items()])

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(('create_repo', repo_id))

    def upload_folder(self, folder_path, allow_patterns, **kwargs):
        import os
        files = sorted(f for f in os.listdir(folder_path)
                       if any(fnmatch.fnmatch(f, p) for p in allow_patterns))
        self.calls.append(('upload_folder', files))
        return types.SimpleNamespace(oid='folder-commit')

    def upload_file(self, path_or_fileobj, path_in_repo, **kwargs):
        self.calls.append(('upload_file', path_in_repo,
                           path_or_fileobj.decode()))
        return types.SimpleNamespace(oid='card-commit')

    def create_tag(self, repo_id, tag, repo_type=None, revision=None,
                   exist_ok=False):
        self.calls.append(('create_tag', tag, revision, exist_ok))


@pytest.fixture
def bundle(tmp_path):
    folder = tmp_path / 'bundle'
    folder.mkdir()
    for name in BUNDLE + ['optimizer_state.pt']:
        (folder / name).write_text('x')
    (folder / 'README.md').write_text('# local notes')
    return folder


@pytest.fixture
def api(monkeypatch):
    holder = {}

    def make(tags=None, exists=True):
        holder['api'] = FakeApi(tags, exists)
        monkeypatch.setattr(
            hub, '_require_hub',
            lambda: types.SimpleNamespace(HfApi=lambda token=None: holder['api']))
        return holder['api']
    return make


def test_card_is_uploaded_last_and_folder_readme_is_skipped(bundle, api):
    fake = api(exists=False)
    hub.push_model_to_hub(bundle, 'org/TERRA-test', corpus='Corpus-X',
                          tag='v1.0')
    names = [call[0] for call in fake.calls]
    assert names == ['create_repo', 'upload_folder', 'upload_file',
                     'create_tag']
    # Bundle files only: no README.md, no training leftovers
    assert fake.calls[1][1] == sorted(BUNDLE)
    # The generated card (not the folder's README.md) is the last upload
    assert fake.calls[2][1] == 'README.md'
    assert 'Corpus-X' in fake.calls[2][2]
    # The tag points at exactly that revision and is not 'exist_ok'
    assert fake.calls[3] == ('create_tag', 'v1.0', 'card-commit', False)


def test_custom_model_card(bundle, api):
    fake = api()
    hub.push_model_to_hub(bundle, 'org/TERRA-test', model_card='# My card')
    assert fake.calls[-1] == ('upload_file', 'README.md', '# My card')


def test_existing_tag_is_refused_before_uploading(bundle, api):
    fake = api(tags={'v1.0': 'abcdef1234567890'})
    with pytest.raises(ValueError, match="Tag 'v1.0' already exists"):
        hub.push_model_to_hub(bundle, 'org/TERRA-test', tag='v1.0')
    assert fake.calls == []
    # A new tag is fine
    hub.push_model_to_hub(bundle, 'org/TERRA-test', tag='v1.0.1')
    assert fake.calls[-1] == ('create_tag', 'v1.0.1', 'card-commit', False)


def test_cli_model_card(bundle, api, tmp_path):
    fake = api()
    card = tmp_path / 'card.md'
    card.write_text('# From file')
    hub.main(['upload', '--folder', str(bundle), '--repo-id', 'org/TERRA-test',
              '--model-card', str(card)])
    assert fake.calls[-1] == ('upload_file', 'README.md', '# From file')
