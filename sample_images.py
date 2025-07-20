import os
import shutil
import random

source_dir = "original_dataset"
dest_root = "data"
splits = {
    'train': 500,
    'val': 100,
    'test': 100
}
categories = ['Healthy', 'Coccidiosis', 'Salmonella', 'NewCastle']

for cat in categories:
    files = os.listdir(os.path.join(source_dir, cat))
    random.shuffle(files)
    split1 = splits['train']
    split2 = splits['train'] + splits['val']

    train_files = files[:split1]
    val_files = files[split1:split2]
    test_files = files[split2:split2+splits['test']]

    for fname, setname in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        out_dir = os.path.join(dest_root, setname, cat)
        os.makedirs(out_dir, exist_ok=True)
        for f in fname:
            shutil.copy(os.path.join(source_dir, cat, f), os.path.join(out_dir, f))

print("Sampled images distributed to train/val/test folders.")