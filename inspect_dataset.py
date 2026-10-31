
from pathlib import Path

# Locate the ARFF dataset
dataset_path = (
    Path(__file__).parent
    / "dataset"
    / "Training Dataset.arff"
)

if not dataset_path.exists():
    print("Dataset file not found!")
else:
    print("Dataset found successfully!\n")

    with open(dataset_path, "r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    print("----- DATASET INFORMATION -----")

    print("\nRelation:")
    for line in lines:
        if line.strip().lower().startswith("@relation"):
            print(line.strip())

    print("\nAttributes:")
    for line in lines:
        if line.strip().lower().startswith("@attribute"):
            print(line.strip())

    print("\nFirst 10 data rows:")
    data_started = False
    count = 0

    for line in lines:
        if line.strip().lower() == "@data":
            data_started = True
            continue

        if data_started and line.strip() and not line.strip().startswith("%"):
            print(line.strip())
            count += 1

            if count == 10:
                break