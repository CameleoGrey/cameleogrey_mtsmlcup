
import numpy as np
from pathlib import Path

from classes.utils import *
from classes.paths_config import *
from classes.scrapping.GreyHtmlContentParser import *

content_parser = GreyHtmlContentParser()

"""html_example_name = "small_4"
#html_example_name = "big"
#html_example_name = "broken"
html_path = Path( data_dir, "html_examples", "{}.html".format( html_example_name ) )
html_text = read_html( html_path )
parsed_content = parse_useful_content( html_text )"""

html_dir = Path( data_dir, "scrapped_htmls" )
url_content_dict = content_parser.parse_html_dir(html_dir, n_jobs=14)
save( url_content_dict, Path(interim_dir, "url_content_dict.pkl") )
    
print("done")