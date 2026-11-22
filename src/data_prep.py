import pandas as pd

def load_and_inspect(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}\n")
    print(f"\nDataset columns:\n\n {df.columns.tolist()}\n")
    print(f"\nTrue values:\n \n{df.isna().sum()}\n")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    return df

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    print(f"{df['Machine failure'].value_counts(normalize=True)*100}")


if __name__ =="__main__":
    df=load_and_inspect("data/ai4i2020.csv")
