from datasets import load_from_disk
import pickle

# Paths
ID ='human_cohort2_60m_32_None_None_None_gene_corrected_read_depth_None_shifted_log_knn_10'
data_path = f'/lustre/scratch126/cellgen/team361/DATASETS/gold/cell-graph-tokenizer/human_cohort2_60m/{ID}.dataset/'
train_indices_path = f'pickle_folder/precomputed_split_{ID}_train.pkl'
validation_indices_path = f'pickle_folder/precomputed_split_{ID}_validation.pkl'
test_indices_path = f'pickle_folder/precomputed_split_{ID}_test.pkl'
train_subset_indices_path = f'pickle_folder/precomputed_split_{ID}_10m_train_subset.pkl'
# Load the Hugging Face dataset
dataset = load_from_disk(data_path)

# Load split indices
with open(train_indices_path, 'rb') as f:
    train_indices = pickle.load(f)

with open(validation_indices_path, 'rb') as f:
    validation_indices = pickle.load(f)

with open(test_indices_path, 'rb') as f:
    test_indices = pickle.load(f)

with open(train_subset_indices_path, 'rb') as f:
    train_subset_indices = pickle.load(f)

# Helper function to save n_nonzero_tokens
def save_n_nonzero_tokens(dataset, indices, output_path):
    # Select entries based on indices
    selected_data = dataset.select(indices)
    # Extract 'n_nonzero_tokens' and save as a list
    n_nonzero_tokens = selected_data['n_nonzero_tokens']
    print(len(n_nonzero_tokens))
    with open(output_path, 'wb') as f:
        pickle.dump(n_nonzero_tokens, f)

# Save n_nonzero_tokens for train, validation, and test
'''
save_n_nonzero_tokens(dataset, validation_indices, f'pickle_folder/n_nonzero_tokens_{ID}_validation.pkl')
print('done1')
save_n_nonzero_tokens(dataset, train_indices, f'pickle_folder/n_nonzero_tokens_{ID}_train.pkl')
print('done2')
save_n_nonzero_tokens(dataset, test_indices, f'pickle_folder/n_nonzero_tokens_{ID}_test.pkl')
print('done3')
'''
save_n_nonzero_tokens(dataset, train_subset_indices, f'pickle_folder/n_nonzero_tokens_{ID}_10m_train_subset.pkl')

print("Saved n_nonzero_tokens for train, validation, and test datasets.")

