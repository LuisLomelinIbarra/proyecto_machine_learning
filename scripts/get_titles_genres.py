import pandas as pd
import constant

# Used columns
cols_basics = [
    'tconst',
    'titleType',
    'primaryTitle',
    'originalTitle',
    'isAdult',
    'startYear',
    'runtimeMinutes',
    'genres'
]

cols_akas = [
    'titleId',
    'language',
    'isOriginalTitle'
]

cols_ratings = [
    'tconst'
]

cols_genres = [
    'tconst',
    'genre'
]

# Data types definitions
dtypes_basics = {
    'titleType': 'str',
    'isAdult': 'str',
    'startYear': 'str',
    'runtimeMinutes': 'str'
}

dtypes_akas = {
    'isOriginalTitle': 'str',
}

# Import IMDb title basics dataset
title_basics = pd.read_csv(constant.TITLE_BASICS_FILE, encoding='utf-8',
                           sep='\t', header=0, usecols=cols_basics, dtype=dtypes_basics)

# Import IMDb title akas dataset
title_akas = pd.read_csv(constant.TITLE_AKAS_FILE, encoding='utf-8',
                         sep='\t', header=0, dtype=dtypes_akas, usecols=cols_akas)

# Import IMDb title ratings dataset
title_ratings = pd.read_csv(constant.TITLE_RATINGS_FILE, encoding='utf-8',
                            sep='\t', header=0, usecols=cols_ratings)

# Rename start year column to 'year'
title_basics = title_basics.rename(columns={'startYear': 'year'})

# Select entries whose title type is movie
# Select entries whose year and runtime minutes are known
title_basics = title_basics.loc[((title_basics['titleType'] == constant.TITLE_TYPE_MOVIE)
                                 & (title_basics['year'] != '\\N')
                                 & (title_basics['runtimeMinutes'] != '\\N'))]

# Select entries that have been rated
title_basics = title_basics[title_basics['tconst'].isin(
    title_ratings['tconst'])]

# Select entries whose language is known
title_akas = title_akas.loc[(title_akas['language'] != '\\N')]

# Cast year data type from string to int
title_basics['year'] = title_basics['year'].astype('int64')

# Select entries whose start year >= 2000 and sort in ascending order
title_basics = title_basics.sort_values(
    by=['year']).loc[title_basics['year'] >= constant.YEAR]

# Genres
genres_list = []

for index, row in title_basics.iterrows():
    genres_parsed = row['genres'].split(',')

    for g in genres_parsed:
        genres_list.append([row['tconst'], g])

genres_df = pd.DataFrame(genres_list, columns=cols_genres)

# Merge titles basics and akas into a single dataset
titles = pd.merge(title_basics, title_akas, how='inner',
                  left_on='tconst', right_on='titleId')

# Drop duplicates
titles = titles.drop_duplicates(subset=['tconst'])

# Drop unused columns
titles = titles.drop('titleId', 1)

# Export filtered dataset
titles.to_csv(constant.TITLES_FILE, index=False)

genres_df.to_csv(constant.GENRES_FILE, index=False)
