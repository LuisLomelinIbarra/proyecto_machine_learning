import pandas as pd
import constant

# Used columns
cols_cast = [
    'tconst',
    'nconst'
]

cols_names = [
    'nconst',
    'primaryName',
    'birthYear'
]

# Data types definitions
dtypes_cast = {
    'tconst': 'str'
}

dtypes_names = {
    'nconst': 'str',
    'primaryName': 'str',
    'birthYear': 'str'
}

# Import filtered cast dataset
cast = pd.read_csv(constant.CAST_FILE, encoding='utf-8',
                   header=0, usecols=cols_cast, dtype=dtypes_cast)

# Import IMDb name basics dataset
name_basics = pd.read_csv(constant.NAME_BASICS_FILE,
                          encoding='utf-8', sep='\t', header=0, usecols=cols_names)

# Select entries that have a related movie in the filtered titles dataset
names = name_basics[name_basics['nconst'].isin(cast['nconst'])]

names.to_csv(constant.NAMES_FILE, index=False)
