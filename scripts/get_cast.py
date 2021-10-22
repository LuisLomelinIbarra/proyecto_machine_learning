import pandas as pd
import constant

# Used columns
cols_titles = [
    'tconst'
]

cols_principals = [
    'tconst',
    'nconst',
    'category'
]

# Data types definitions
dtypes_titles = {
    'tconst': 'str'
}

# Import filtered titles dataset
titles = pd.read_csv(constant.TITLES_FILE, encoding='utf-8',
                     header=0, usecols=cols_titles, dtype=dtypes_titles)

# Import IMDb title principals dataset
title_principals = pd.read_csv(constant.TITLE_PRINCIPALS_FILE,
                               encoding='utf-8', sep='\t', header=0, usecols=cols_principals)

# Select entries that have a related movie in the filtered titles dataset
cast = title_principals[title_principals['tconst'].isin(titles['tconst'])]

cast.to_csv(constant.CAST_FILE, index=False)
