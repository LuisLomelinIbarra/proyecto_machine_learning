import pandas as pd
import constant

# Used columns
cols_titles = [
    'tconst'
]

# Data types definitions
dtypes_titles = {
    'tconst': 'str'
}

# Import filtered titles dataset
titles = pd.read_csv(constant.TITLES_FILE, encoding='utf-8',
                     header=0, usecols=cols_titles, dtype=dtypes_titles)

# Import IMDb titles ratings dataset
title_ratings = pd.read_csv(constant.TITLE_RATINGS_FILE,
                            encoding='utf-8', sep='\t', header=0)

# Select entries that have a related movie in the filtered titles dataset
ratings = title_ratings[title_ratings['tconst'].isin(titles['tconst'])]

# Export filtered dataset
ratings.to_csv(constant.RATINGS_FILE, index=False)
