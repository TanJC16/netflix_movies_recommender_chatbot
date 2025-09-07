def clean_data(input: str):
    import pandas as pd
    
    df = pd.read_csv(input)
    
    # Drop Irrelevant Columns
    columns_to_drop = ["show_id", "type", "duration", "budget", "revenue", "vote_average"]
    df = df.drop(columns_to_drop, axis=1)
    
    # Remove Duplicate
    df.drop_duplicates(inplace=True)
    
    # Standardize Data
    df['title'] = df['title'].str.strip()
    df['genres'] = df['genres'].str.lower()
    df['genres'] = df['genres'].str.split(", ")
    df['director'] = df['director'].str.split(", ")
    df['cast'] = df['cast'].str.split(", ")
    
    # Fill null
    df['country'].fillna("Unknown", inplace=True)
    df['description'].fillna("No description available", inplace=True)
    
    # Drop null
    df = df.dropna(subset=["genres", "cast", "director"]) # 2.2% will gone
    
    df.to_csv("cleaned_movies.csv", index=False)

clean_data('netflix_movies_detailed_up_to_2025.csv')