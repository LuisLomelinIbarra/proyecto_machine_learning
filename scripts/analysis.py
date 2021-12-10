import pandas as pd
import constant

# Titles attributes
cols_titles = [
    'language'
]

col_titles = 'language'

# Genres attributes
cols_genres = [
    'genre'
]

col_genres = 'genre'

cols_cast = [
    'category'
]

col_cast = 'category'

cols_names = [
    'nconst'
]

col_names = 'nconst'

# Data types definitions
dtypes_titles = {
    'language': 'str'
}

dtypes_genres = {
    'genre': 'str'
}

dtypes_cast = {
    'category': 'str'
}

dtypes_names = {
    'nconst': 'str'
}

# Import filtered titles dataset
titles = pd.read_csv(constant.TITLES_FILE, encoding='utf-8',
                     header=0, usecols=cols_titles, dtype=dtypes_titles)

# Import filtered genres dataset
genres = pd.read_csv(constant.GENRES_FILE, encoding='utf-8',
                     header=0, usecols=cols_genres, dtype=dtypes_genres)

# Import filtered cast dataset
cast = pd.read_csv(constant.CAST_FILE, encoding='utf-8',
                   header=0, usecols=cols_cast, dtype=dtypes_cast)

# Import filtered names dataset
names = pd.read_csv(constant.NAMES_FILE, encoding='utf-8',
                    header=0, usecols=cols_names, dtype=dtypes_names)

print(len(titles.index))
print(len(genres.index))
print(len(cast.index))
print(len(names.index))
