import pandas as pd
import constant

# Used columns
cols_titles = [
    'tconst'
]

cols_directors = [
    'tconst',
    'nconst'
]

cols_writers = [
    'tconst',
    'nconst'
]

# Data types definitions
dtypes_titles = {
    'tconst': 'str'
}

# Import filtered titles dataset
titles = pd.read_csv(constant.TITLES_FILE, encoding='utf-8',
                     header=0, usecols=cols_titles, dtype=dtypes_titles)

# Import IMDb title crew dataset
title_crew = pd.read_csv(constant.TITLE_CREW_FILE,
                         sep='\t', encoding='utf-8', header=0)

# Select entries that have a related movie in the filtered titles dataset
crew = title_crew[title_crew['tconst'].isin(titles['tconst'])]

directors_list = []
writers_list = []

for index, row in crew.iterrows():
    directors_parsed = row['directors'].split(',')
    writers_parsed = row['writers'].split(',')

    for d in directors_parsed:
        directors_list.append([row['tconst'], d])

    for w in writers_parsed:
        writers_list.append([row['tconst'], d])

directors_df = pd.DataFrame(directors_list, columns=cols_directors)
writers_df = pd.DataFrame(writers_list, columns=cols_writers)

directors_df.to_csv(constant.DIRECTORS_FILE, index=False)
writers_df.to_csv(constant.WRITERS_FILE, index=False)
