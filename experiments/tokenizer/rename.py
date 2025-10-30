import os

folder = '/data13/datasets/pretrain/ultrafineweb-en_json/'

files = [f for f in os.listdir(folder)]

for f in files:
    old_path = os.path.join(folder, f)
    new_path = old_path
    new_path = new_path.replace('-of-2048', '').replace('ultrafineweb_en_ultrafineweb-en-', '')
    os.rename(old_path, new_path)
