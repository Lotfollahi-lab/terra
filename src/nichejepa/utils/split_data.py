import pickle
#import tqdm

# Configuration for batch IDs

val_batch_ids = [
    '1000_batch23', '1000_batch24', '1000_batch25', '1000_batch28', '1000_batch29'
]
test_batch_ids = [
    '1000_batch4', '1000_batch6', '1000_batch11', '1000_batch12', '1000_batch18',
    '1000_batch19', '1000_batch21', '1000_batch22', '1000_batch34', '1000_batch38',
    '1000_batch20', '1000_batch26', '1000_batch27', '1000_batch32', '1000_batch36',
    '1002_batch18', '1002_batch46', '1002_batch31', '1002_batch34', '1002_batch23',
    '1002_batch53', '1002_batch55', '1002_batch16', '1002_batch39', '1002_batch3',
    '1002_batch26', '1002_batch45', '1002_batch27', '1002_batch40',
    '1002_batch4', '78_batch3', '78_batch10', '78_batch1', '78_batch0', '78_batch2',
    '78_batch6', '78_batch7', '78_batch9', '78_batch8', '78_batch4', '78_batch5',
    '59_batch0', '59_batch3', '59_batch6', '59_batch7', '59_batch2', '59_batch4',
    '59_batch1', '59_batch5','49_batch0', '1007_batch0', '1007_batch1', '1007_batch2',
	'1007_batch3', '1007_batch4', '1007_batch5', '1007_batch6', '1007_batch7',
]

'''
import random

# Prefixes to use for generating random items
prefixes = [
    '23_batch0', '1000_batch24', '1000_batch25', '1000_batch28', '1000_batch29',
    '1000_batch4', '1001_batch4', '1021_batch41'
]
random_items = [
    f"{prefix}_{random.randint(100, 9999)}"
    for prefix in prefixes
    for _ in range(100)
]

val_batch_ids = [
    '23_batch0', '1000_batch24', '1000_batch25', '1000_batch28', '1000_batch29'
]
test_batch_ids = [
    '1000_batch4'
]

# Load cell_ids from pickle file
with open("cell_ids.pkl", "rb") as f:
    cell_ids = pickle.load(f)
cell_ids = cell_ids[0:1000]
cell_ids = random_items
'''
ID ='human_cohort2_60m_32_None_None_None_gene_corrected_read_depth_None_shifted_log_knn_10'
with open(f"pickle_folder/cell_ids_{ID}_.pkl", "rb") as f:
    cell_ids = pickle.load(f)

#cell_ids = cell_ids[0:1000]
# Initialize lists
validation_indices = []
test_indices = []

print('done0')
print(len(cell_ids))

# Single loop to determine validation and test indices
for index, cell_id in enumerate(cell_ids):
    if any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}" for batch_id in val_batch_ids):
        validation_indices.append(index)
    elif any(batch_id == f"{cell_id.split('_')[0]}_{cell_id.split('_')[1]}" for batch_id in test_batch_ids):
        test_indices.append(index)
print('done1')
# Another loop to determine train indices
excluded_indices = set(validation_indices).union(test_indices)
print('done2')
print(len(excluded_indices))
print(len(validation_indices)+len(test_indices))
print(len(validation_indices))
print(len(test_indices))


# Compute train_indices
train_indices = [index for index in range(len(cell_ids)) if index not in excluded_indices]
'''
train_indices = [
    index for index in range(len(cell_ids)) if index not in validation_indices and index not in test_indices
]
'''
print('done3')
# Extract unique identifiers (first and second part before the first underscore) for each index list and store in sets
train_unique = {f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in train_indices}
validation_unique = {f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in validation_indices}
test_unique = {f"{cell_ids[i].split('_')[0]}_{cell_ids[i].split('_')[1]}" for i in test_indices}

# Print the results
print("Train unique identifiers:", train_unique)
print("Validation unique identifiers:", validation_unique)
print("Test unique identifiers:", test_unique)

print('done4')

# Save the splits as pickle files
with open(f"pickle_folder/precomputed_split_{ID}_validation.pkl", "wb") as f:
    pickle.dump(validation_indices, f)

with open(f"pickle_folder/precomputed_split_{ID}_test.pkl", "wb") as f:
    pickle.dump(test_indices, f)

with open(f"pickle_folder/precomputed_split_{ID}_train.pkl", "wb") as f:
    pickle.dump(train_indices, f)

print("Splits saved as precomputed_split_60m_validation.pkl, precomputed_split_60m_test.pkl, and precomputed_split_60m_train.pkl")
