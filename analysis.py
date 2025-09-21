import pandas as pd
import glob
import os

EXCEL_IMPORT_DIR = "./data/excel/"

def read_excels():
    try:
        print("Reading Excel files...")
        paths = glob.glob(os.path.join(EXCEL_IMPORT_DIR, "**", "*.xlsx"), recursive=True)
        docs = []
        print(paths)
        for p in paths:
            try:
                # Read the metadata first
                metadata = pd.read_excel(p, nrows=1, header=None)
                metadata_text = metadata.to_string(index=False, header=False)

                # Read the actual data now
                df = pd.read_excel(p, skiprows=1)
                docs.append((p, df))
                print(f"Read {p} with shape {df.shape}")
            except Exception as e:
                print(f"Error reading {p}: {e}")

        chunks = []
        chunks.append(metadata_text)
        for i, row in df.iterrows():
            chunk = f"Country: {row['Country']}, " + \
                    f"Export in Billion USD for Fiscal Year 2023-2024: {row['2023-2024']}, " + \
                    f"Export in Billion USD for Fiscal Year 2024-2025: {row['2024-2025']}"
            print(f"Processing row {i}: {chunk}")
            chunks.append(chunk)
        print(f"Total chunks created: {len(chunks)}")
        chunks.append("End of this Document.")
        return chunks
    except Exception as e:
        print(f"Error in read_excels: {e}")

def main():
    read_excels()

if __name__ == "__main__":
    main()
