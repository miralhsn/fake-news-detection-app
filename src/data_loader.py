import pandas as pd
import os

def load_data(fake_path=None, real_path=None):
    """
    Load and merge fake and real news datasets.
    
    Args:
        fake_path (str): Path to the fake news CSV file
        real_path (str): Path to the real news CSV file
    
    Returns:
        pd.DataFrame: Combined and shuffled dataset with 'text' and 'label' columns
    """
    # Always resolve paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if fake_path is None:
        fake_path = os.path.join(base_dir, 'data', 'Fake.csv')
    if real_path is None:
        real_path = os.path.join(base_dir, 'data', 'Real.csv')
    
    # Check if files exist
    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        raise FileNotFoundError(f"One or both dataset files not found. Looked for: {fake_path} and {real_path}")
    
    # Load datasets
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    
    # Add labels
    fake['label'] = 0  # 0 for fake news
    real['label'] = 1  # 1 for real news
    
    # Combine datasets
    df = pd.concat([fake, real], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df 