import re

with open("eval_output.txt") as f:
    lines = f.readlines()

mismatches = 0
ranking_errors = 0
content_errors = 0

rust_tokens = []
py_top1 = None

for line in lines:
    line = line.strip()
    if line.startswith("Mismatch at step"):
        mismatches += 1
    elif line.startswith("Rust Top 3:"):
        # extract numbers right after '[' and ' '
        matches = re.findall(r'(\d+) \(val=', line)
        rust_tokens = [int(m) for m in matches]
    elif line.startswith("PyTorch Top 3:"):
        matches = re.findall(r'(\d+) \(val=', line)
        if matches:
            py_top1 = int(matches[0])
            if py_top1 in rust_tokens:
                ranking_errors += 1
            else:
                content_errors += 1

print(f"Total Mismatches Analyzed: {mismatches}")
print(f"Ranking Errors (PyTorch top-1 IS in Rust top-3): {ranking_errors}")
print(f"Content Errors (PyTorch top-1 IS NOT in Rust top-3): {content_errors}")
