import hashlib

def get_grouped_split(s1_ids, test_size=0.2, seed=42):
    """
    Deterministically splits a list of S1 entity IDs into train and validation sets.
    Uses MD5 hashing to ensure that the split is stable across runs, machines, and orderings.
    
    Args:
        s1_ids: Iterable of S1 entity IDs.
        test_size: The proportion of IDs to include in the validation split (0.0 to 1.0).
        seed: An integer seed to perturb the hash if different splits are needed.
        
    Returns:
        tuple: (train_ids, val_ids) both as lists.
    """
    train_ids = []
    val_ids = []
    
    # 0xFFFFFFFF is maximum 32-bit unsigned integer
    max_hash_val = 0xFFFFFFFF
    threshold = test_size * max_hash_val
    
    for s1_id in sorted(s1_ids):
        # Create a deterministic string to hash
        hash_input = f"{s1_id}_{seed}".encode('utf-8')
        # Use md5 and take first 8 hex characters (32 bits)
        hash_hex = hashlib.md5(hash_input).hexdigest()[:8]
        hash_val = int(hash_hex, 16)
        
        if hash_val < threshold:
            val_ids.append(s1_id)
        else:
            train_ids.append(s1_id)
            
    return train_ids, val_ids
