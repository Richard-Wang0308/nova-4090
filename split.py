import csv
import sys
import os

def split_csv(input_file):
    # Define output files and their writers
    output_files = {}
    writers = {}
    header = None

    # Open all 5 output files
    for i in range(1, 6):
        f = open(f"data/rxn{i}.csv", "w", newline="")
        output_files[i] = f
        writers[i] = csv.writer(f)

    try:
        with open(input_file, "r", newline="") as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read header row

            # Write header to all output files
            for w in writers.values():
                w.writerow(header)

            matched   = 0
            unmatched = 0

            for row in reader:
                if not row:
                    continue  # Skip empty lines

                molecule_name = row[0]
                assigned = False

                for i in range(1, 6):
                    if molecule_name.startswith(f"rxn:{i}:"):
                        writers[i].writerow(row)
                        matched += 1
                        assigned = True
                        break

                if not assigned:
                    print(f"[WARNING] Unmatched row skipped: {row[0]}")
                    unmatched += 1

    finally:
        for f in output_files.values():
            f.close()

    print(f"\n✅ Done! Split '{input_file}' into rxn1.csv ~ rxn5.csv")
    print(f"   Matched rows   : {matched}")
    if unmatched:
        print(f"   Unmatched rows : {unmatched} (skipped)")

    # Print row counts per file
    print("\n📄 Row counts per output file (excluding header):")
    for i in range(1, 6):
        fname = f"data/rxn{i}.csv"
        with open(fname, "r") as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        print(f"   {fname}: {row_count} rows")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python split.py <input_csv_file>")
    #     sys.exit(1)

    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/mols.csv'

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    split_csv(input_file)