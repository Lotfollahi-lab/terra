import random
import pickle
from collections import defaultdict

# Example data
with open("cell_ids_60m.pkl", "rb") as f:
    cell_ids = pickle.load(f)

val_batch_ids = [
    '1000_batch23', '1000_batch24', '1000_batch25', '1000_batch28', '1000_batch29',
    '1000_batch4', '1000_batch6', '1000_batch11', '1000_batch12', '1000_batch18',
    '1000_batch19', '1000_batch21', '1000_batch22', '1000_batch34', '1000_batch38',
    '1000_batch20', '1000_batch26', '1000_batch27', '1000_batch32', '1000_batch36',
    '1002_batch18', '1002_batch46', '1002_batch31', '1002_batch34', '1002_batch23',
    '1002_batch53', '1002_batch55', '1002_batch16', '1002_batch39', '1002_batch3',
    '1002_batch26', '1002_batch45', '1002_batch27', '1002_batch40',
    '1002_batch4', '78_batch3', '78_batch10', '78_batch1', '78_batch0', '78_batch2',
    '78_batch6', '78_batch7', '78_batch9', '78_batch8', '78_batch4', '78_batch5',
    '59_batch0', '59_batch3', '59_batch6', '59_batch7', '59_batch2', '59_batch4',
    '59_batch1', '59_batch5', '49_batch0', '1007_batch0', '1007_batch1', '1007_batch2',
    '1007_batch3', '1007_batch4', '1007_batch5', '1007_batch6', '1007_batch7',
]

# Step 1: Exclude items with prefixes in val_batch_ids
def extract_prefix(item):
    return item.rsplit("_", 1)[0]  # Extract prefix, e.g., "23_batch0_0" -> "23_batch0"

# Filtering the items and keeping track of indices
filtered_items = [
    (index, item)  # Keep track of the original index
    for index, item in enumerate(cell_ids)
    if extract_prefix(item) not in val_batch_ids
]

# Step 2: Group items by prefix
grouped_items = defaultdict(list)
for index, item in filtered_items:
    prefix = extract_prefix(item)
    grouped_items[prefix].append((index, item))

# Step 3: Stratified sampling
train_subset_indices = []
remaining_items_indices = []

# Fixed random seed for reproducibility
random.seed(42)

for prefix, group in grouped_items.items():
    random.shuffle(group)  # Shuffle within the prefix
    split_idx = int(len(group) * 0.19)  # 19% split
    train_subset_indices.extend([index for index, _ in group[:split_idx]])  # Add indices of 19% to train_subset
    remaining_items_indices.extend([index for index, _ in group[split_idx:]])  # Add indices of remaining items

# Outputs
print("Train Subset Indices (19%):", len(train_subset_indices))
print("Remaining Items Indices (81%):", len(remaining_items_indices))

# Save results to pickle files
with open("precomputed_split_60m_10m_train_subset_indices.pkl", "wb") as f:
    pickle.dump(train_subset_indices, f)
