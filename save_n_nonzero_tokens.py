from datasets import load_from_disk
import pickle

# Paths
data_path = '/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/human_cohort1_50m/human_cohort1_50m_32_None_shifted_log_knn_10.dataset/'
train_indices_path = 'precomputed_split_train.pkl'
validation_indices_path = 'precomputed_split_validation.pkl'
test_indices_path = 'precomputed_split_test.pkl'

# Load the Hugging Face dataset
dataset = load_from_disk(data_path)

# Load split indices
with open(train_indices_path, 'rb') as f:
    train_indices = pickle.load(f)

with open(validation_indices_path, 'rb') as f:
    validation_indices = pickle.load(f)

with open(test_indices_path, 'rb') as f:
    test_indices = pickle.load(f)

# Helper function to save n_nonzero_tokens
def save_n_nonzero_tokens(dataset, indices, output_path):
    # Select entries based on indices
    selected_data = dataset.select(indices)
    # Extract 'n_nonzero_tokens' and save as a list
    n_nonzero_tokens = selected_data['n_nonzero_tokens']
    with open(output_path, 'wb') as f:
        pickle.dump(n_nonzero_tokens, f)

# Save n_nonzero_tokens for train, validation, and test
save_n_nonzero_tokens(dataset, validation_indices, 'n_nonzero_tokens_validation.pkl')
print('done1')
save_n_nonzero_tokens(dataset, train_indices, 'n_nonzero_tokens_train.pkl')
print('done2')
save_n_nonzero_tokens(dataset, test_indices, 'n_nonzero_tokens_test.pkl')
print('done3')

print("Saved n_nonzero_tokens for train, validation, and test datasets.")

