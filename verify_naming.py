import os
import glob

root = r"C:\AML_data\SPair-71k"
split = "trn"
split_file = os.path.join(root, "Layout", "large", f"{split}.txt")

with open(split_file, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

sample_line = lines[0]
print(f"Sample line: {sample_line}")

target_filename = sample_line.replace(":", "_") + ".json"
print(f"Expected filename: {target_filename}")

# Search for it
matches = glob.glob(os.path.join(root, "**", f"{target_filename}"), recursive=True)
print(f"Found matches: {matches}")
