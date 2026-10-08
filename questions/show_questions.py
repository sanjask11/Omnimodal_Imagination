import json, csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "questions/questions_p2.json"
items = json.load(open(path))["items"]

# 1. print for review
for i, it in enumerate(items, 1):
    role = it.get("target_role") or "-"
    print("=" * 70)
    print(f"Q{i} | {it['task']} | role: {role} | GT: {it['answer']}")
    print(it["question"])

# 2. write rating sheet
out = path.replace(".json", "_rating_sheet.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "task", "question", "sanjana (1-5)", "tibo (1-5)", "nazia (1-5)", "notes"])
    for i, it in enumerate(items, 1):
        w.writerow([f"Q{i}", it["task"], it["question"], "", "", "", ""])
print(f"\nwrote {out}")