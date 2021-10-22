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

titles_list = titles[col_titles].drop_duplicates().to_list()
titles_dict = {i: titles_list[i] for i in range(0, len(titles_list))}
titles_df = pd.DataFrame.from_dict(
    titles_dict, orient='index', columns=['key'])
titles_df.to_csv(constant.TITLES_DICT, index=True, index_label='value')

genres_list = genres[col_genres].drop_duplicates().to_list()
genres_dict = {i: genres_list[i] for i in range(0, len(genres_list))}
genres_df = pd.DataFrame.from_dict(
    genres_dict, orient='index', columns=['key'])
genres_df.to_csv(constant.GENRES_DICT, index=True, index_label='value')

cast_list = cast[col_cast].drop_duplicates().to_list()
cast_dict = {i: cast_list[i] for i in range(0, len(cast_list))}
cast_df = pd.DataFrame.from_dict(
    cast_dict, orient='index', columns=['key'])
cast_df.to_csv(constant.CAST_DICT, index=True, index_label='value')

names_list = names[col_names].drop_duplicates().to_list()
names_dict = {i: names_list[i] for i in range(0, len(names_list))}
names_df = pd.DataFrame.from_dict(
    names_dict, orient='index', columns=['key'])
names_df.to_csv(constant.NAMES_DICT, index=True, index_label='value')
