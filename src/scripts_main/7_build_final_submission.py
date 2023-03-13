
import pandas as pd
from pathlib import Path

from classes.paths_config import *

age_submission = pd.read_csv( Path(production_dir, "age_predicts.csv") )
is_male_submission = pd.read_csv( Path(production_dir, "is_male_predicts.csv") )

final_submission = age_submission.merge( is_male_submission, on="user_id", how="inner" )
final_submission = final_submission[["user_id", "age", "is_male"]]
final_submission.to_csv( Path(production_dir, "my_submission.csv"), index=False )

print("done")