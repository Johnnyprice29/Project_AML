import os
import glob
import json

root = r"C:\AML_data\SPair-71k"
print(f"Root exists: {os.path.exists(root)}")

# Check top level
print("Folders in SPair-71k:", os.listdir(root))

# Check ImageAnnotation
ann_dir = os.path.join(root, "ImageAnnotation")
cats = os.listdir(ann_dir)
print(f"Categories in ImageAnnotation: {cats[:5]}")

files = os.listdir(os.path.join(ann_dir, cats[0]))
print(f"Files in {cats[0]}: {files[:5]}")

# Check PairAnnotation (if exists)
pair_ann_dir = os.path.join(root, "PairAnnotation")
if os.path.exists(pair_ann_dir):
    print("Folders in PairAnnotation:", os.listdir(pair_ann_dir))
else:
    print("PairAnnotation folder does NOT exist.")

# Check Layout/large/trn.txt
trn_txt = os.path.join(root, "Layout", "large", "trn.txt")
with open(trn_txt, "r") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Sample trn line: {lines[0]}")

# Try to find where the JSON for the sample trn line is
pair_id = lines[0].split(":")[0]
cat = lines[0].split(":")[1]
print(f"Searching for JSON: {pair_id} in category {cat}")

# Recursive search for that pair_id
found = glob.glob(os.path.join(root, "**", f"{pair_id}*"), recursive=True)
print(f"Found matches for {pair_id}: {found}")
