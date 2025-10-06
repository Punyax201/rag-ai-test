import pandas as pd
import glob
import os

EXCEL_IMPORT_DIR = "./data/excel/"

def read_excels(data_dir=EXCEL_IMPORT_DIR):
    try:
        print("Reading Excel files...")
        paths = glob.glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)
        all_chunks = []
        print(paths)
        for p in paths:
            try:
                metadata = pd.read_excel(p, nrows=1, header=None)
                metadata_text = metadata.to_string(index=False, header=False)
                df = pd.read_excel(p, skiprows=1)
                print(f"Read {p} with shape {df.shape}")

                chunks = []
                chunks.append(metadata_text)
                year_cols = [col for col in df.columns if col not in ["Country", "%Growth", "S.No."]]
                for i, row in df.iterrows():
                    chunk = f"Country: {row['Country']}"
                    for year in year_cols:
                        value = row[year] if pd.notna(row[year]) else 0
                        chunk += f", Export in Billion USD for Fiscal Year {year}: {value}"
                    if "%Growth" in df.columns:
                        value = row['%Growth'] if pd.notna(row['%Growth']) else 0
                        chunk += f", Growth Percentage: {value}"
                    print(f"Processing row {i}: {chunk}")
                    chunks.append(chunk)
                chunks.append("End of this Document.")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {p}: {e}")
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    except Exception as e:
        print(f"Error in read_excels: {e}")

def main():
    read_excels()

if __name__ == "__main__":
    main()
