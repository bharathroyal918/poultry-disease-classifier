import os

categories = ['Healthy', 'Coccidiosis', 'Salmonella', 'NewCastle']
sets = ['train', 'val', 'test']

for s in sets:
    for c in categories:
        os.makedirs(f"data/{s}/{c}", exist_ok=True)

print("Folder structure created successfully.")