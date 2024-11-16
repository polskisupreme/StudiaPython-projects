import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Read data from CSV
def read_data(file_path):
    logging.info(f"Reading data from {file_path}")
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data read successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        raise

# Clean data by removing rows with too many missing values and filling others
def clean_data(df):
    logging.info("Rozpoczęto czyszczenie danych.")

    # Zapis oryginalnej liczby wierszy
    original_len = len(df)

    # Usuwanie wierszy z brakującymi wartościami w więcej niż 2 kolumnach
    df.dropna(thresh=5, inplace=True)
    logging.info(f"Usunięto {original_len - len(df)} wierszy z brakującymi danymi.")

    # Uzupełnianie brakujących wartości medianą
    missing_before = df.isnull().sum().sum()

    df['Wiek'].fillna(df['Wiek'].median(), inplace=True)
    df['Średnie Zarobki'].fillna(df['Średnie Zarobki'].median(), inplace=True)

    missing_after = df.isnull().sum().sum()
    filled_missing = missing_before - missing_after
    logging.info(f"Uzupełniono {filled_missing} brakujących wartości.")

    # Standaryzacja kolumn 'Wiek' i 'Średnie Zarobki'
    scaler = StandardScaler()
    df[['Wiek', 'Średnie Zarobki']] = scaler.fit_transform(df[['Wiek', 'Średnie Zarobki']])
    logging.info("Przeprowadzono standaryzację danych.")

    return df

# Standardize numerical columns
def standardize_data(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, num_cols] = scaler.fit_transform(df[num_cols])
    logging.info(f"Standardized numerical columns.")
    return df


# Generate report about data cleaning and standardization
def generate_report(df, df_cleaned, report_file):
    removed_rows = df.shape[0] - df_cleaned.shape[0]
    percent_removed = (removed_rows / df.shape[0]) * 100
    percent_modified = ((df_cleaned.isna().sum().sum()) / df_cleaned.size) * 100

    with open(report_file, 'w') as report:
        report.write(f"Percentage of data removed: {percent_removed:.2f}%\n")
        report.write(f"Percentage of data modified (filled): {percent_modified:.2f}%\n")

    logging.info(f"Report generated at {report_file}")

if __name__ == "__main__":
    # Reading data
    data = read_data('data_student_12345.csv')

    # Cleaning data
    clean_data_df = clean_data(data)

    # Standardizing data
    standardized_data = standardize_data(clean_data_df)

    # Generating report
    generate_report(data, standardized_data, 'report.txt')

    logging.info("Data processing complete.")