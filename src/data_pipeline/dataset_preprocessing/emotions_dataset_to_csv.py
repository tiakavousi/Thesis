import pandas as pd
import os

def main():
    # Define the path to the Excel file.
    excel_path = os.path.join("datasets", "not used", "multiple", "Emotions_GoldSandard_andAnnotation_combined.xlsx")
    
    # Define the output CSV path (same folder as the Excel file)
    output_csv = os.path.join(os.path.dirname(excel_path), "emotions.csv")
    
    # Read all sheets from the Excel file into a dictionary of DataFrames
    sheets_dict = pd.read_excel(excel_path, sheet_name=None)
    
    # Combine all sheets into a single DataFrame
    combined_df = pd.concat(sheets_dict.values(), ignore_index=True)
    
    # Save the combined DataFrame as a CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved as {output_csv}")

if __name__ == "__main__":
    main()
