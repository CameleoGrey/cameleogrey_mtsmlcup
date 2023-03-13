
import os
from pathlib import Path

models_dir = os.path.join("..", "..", "models")
if not Path( models_dir ).exists():
    Path( models_dir ).mkdir(parents=True, exist_ok=True)

data_dir = os.path.join("..", "..", "data")
if not Path( data_dir ).exists():
    Path( data_dir ).mkdir(parents=True, exist_ok=True)

interim_dir = os.path.join( data_dir, "interim" )
if not Path( interim_dir ).exists():
    Path( interim_dir ).mkdir(parents=True, exist_ok=True)

raw_dir = os.path.join( data_dir, "raw" )
if not Path( raw_dir ).exists():
    Path( raw_dir ).mkdir(parents=True, exist_ok=True)

plots_dir = os.path.join( data_dir, "plots" )
if not Path( plots_dir ).exists():
    Path( plots_dir ).mkdir(parents=True, exist_ok=True)

production_dir = os.path.join( data_dir, "production" )
if not Path( production_dir ).exists():
    Path( production_dir ).mkdir(parents=True, exist_ok=True)

scrapped_htmls_dir = os.path.join( data_dir, "scrapped_htmls" )
if not Path( scrapped_htmls_dir ).exists():
    Path( scrapped_htmls_dir ).mkdir(parents=True, exist_ok=True)

competition_data_dir = os.path.join( raw_dir, "data_feather" )