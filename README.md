# DubAir - AirBnB Price Analysis for Dublin, Ireland (Deep Learning Seminar, University of Göttingen, 2022)
![image](https://user-images.githubusercontent.com/58623575/156419875-5149fc5f-6a3c-4e8d-92a4-f19fe4e23b99.png)

This Repo contains several files and folders. In the following a short description is given for the most relevant parts.
In general - the files in the main directory contain the most relevant code for the analysis (Data processing, Models,...)
The "load_data.py" script contains the class which handles the data loading process with several transformations.

**Notice: Notebooks for the image models are not productive as the image data exceeds the capacity of this repo**

## Folders:
- TabNet_selected - gridsearch, model, histories
- TabNet_correlated - gridsearch, model, histories
- archive - files not used anymore for several reasons
- data - originally downloaded data and generated image variables
- img_models - further folders for different image models with saved models, checkpoints, grid search results etc.
- munich - notebooks and csv's concerning munich prediction
- plots - Plots mainly used for Paper
- logs - Several Tensorboard logs to track model training and checkpoints to restore models
- text_data - saved csv's used for load_data.py as well needed Files within Text_Analysis.ipynb
- xgBoost_models - results and models 


## Data generating and similar:
- Imputation_New.ipynb (looking insight variables where imputations were needed)
- OSM.ipynb (work on spatial data)
- Text_Analysis.ipynb (work on text variables)
- image_scraping.ipynb (scraping images - adapted for Munich/Dublin)


## Feature Selection: 
Decorrelate Variables by visual inspection
- feature_selection.ipynb


## Models including grid searches:
- CompoundModel.ipynb
- RoomModel.ipynb
- TabNet_GS_fold_Shap.ipynb (uncorrelated features)
- TabNet_GS_fold_Shap_all.ipynb (correlated features)
- ImageModel.ipynb
- xgBoost.ipynb


## File holding interim results
- StreetData.csv


## Helper files:
- PCA.ipynb (used for experimenting with PCAs during feature_selection.ipynb)
- helpers.py (functions used in several scripts)
- helpers_image_classification.py (functions for image handling in RoomModel.ipynb and image_price.ipynb)
- sentiment_dictionary.csv (NLTK, for Text_Analysis.ipynb)


## Content-making for Paper
- plot_tables_paper.ipynb


## Technical folders or files
.idea, TabNet_GS_logs, __pychache__, .DS_Store and .gitignore 

