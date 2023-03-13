
import os
from pathlib import Path
from classes.paths_config import *
from pprint import pprint

import easyocr

reader = easyocr.Reader(["en", "ru"])

result = reader.readtext(os.path.join(plots_dir, "screenshot_example.png"), 
                         text_threshold = 0.2, contrast_ths = 0.1, canvas_size = 1920)
text_only = []
for i in range(len(result)):
    current_box = result[i]
    current_text = current_box[1]
    text_only.append(current_text)
pprint(text_only)

print("\n\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n\n")

result = reader.readtext(os.path.join(plots_dir, "check_ocr_en.png"), 
                         text_threshold = 0.2, contrast_ths = 0.1, canvas_size = 1920)
text_only = []
for i in range(len(result)):
    current_box = result[i]
    current_text = current_box[1]
    text_only.append(current_text)
pprint(text_only)

print("\n\n\n||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\n\n\n")

result = reader.readtext(os.path.join(plots_dir, "check_ocr_ru.png"),
                         text_threshold = 0.2, contrast_ths = 0.1, canvas_size = 1920)
text_only = []
for i in range(len(result)):
    current_box = result[i]
    current_text = current_box[1]
    text_only.append(current_text)
pprint(text_only)

print("done")