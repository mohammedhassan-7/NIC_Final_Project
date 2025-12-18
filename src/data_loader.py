import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

def load_data():
    df = pd.read_csv('data/cs_papers_api.csv')

    # Process categories
    df['categories'] = df['categories'].str.split()
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])
    X = df['abstract'].fillna('').values

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Dataset loaded: {len(X_train)} training samples, {len(X_val)} validation samples')
    print(f'Number of labels: {y.shape[1]}')
    
    return X_train, X_val, y_train, y_val, mlb, y.shape[1]