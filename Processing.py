# -*- coding: utf-8 -*-
"""
@author: Dylan
"""

#%%

#Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
pd.set_option("display.max_columns",10)

# Images Path
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('input', '*', '*.jpg'))}

lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
        }

#Needs to be made universal across OS & File Structures
tile_df = pd.read_csv('input/HAM10000_metadata.csv')

#%%

tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)

tile_df.describe(exclude=[np.number])

#%%

# Plot data distribution
fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)

#%%

# Loading all images
from skimage.io import imread
tile_df['image'] = tile_df['path'].map(imread)

## see the image size distribution
#tile_df['image'].map(lambda x: x.shape).value_counts()

#%%
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)

#%%

# Average Colour information
rgb_info_df = tile_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v in 
                                  zip(['Red', 'Green', 'Blue'], 
                                      np.mean(x['image'], (0, 1)))}),1)
gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1)
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec
rgb_info_df['Gray_mean'] = gray_col_vec
rgb_info_df.sample(3)

for c_col in rgb_info_df.columns:
    tile_df[c_col] = rgb_info_df[c_col].values
    
sns.pairplot(tile_df[['Red_mean','Green_mean','Blue_mean','Gray_mean','cell_type']],
             hue='cell_type', plot_kws = {'alpha': 0.5})















































