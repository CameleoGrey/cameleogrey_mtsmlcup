

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from classes.paths_config import *
from classes.utils import *
from classes.scrapping.SiteHtmlScrapper import SiteHtmlScrapper

"""url_frequency_dict = load( Path(interim_dir, "url_frequency_dict.pkl") )
uniq_url_hosts = []
for url in url_frequency_dict.keys():
    uniq_url_hosts.append( url )
save( uniq_url_hosts, Path(interim_dir, "uniq_url_hosts.pkl") )"""


"""options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
browser = webdriver.Chrome(options=options)
#browser.get('http://www.google.com/')
browser.get('http://www.ya.ru/')
browser.save_screenshot(Path(plots_dir, "screenshot_example.png"))"""

# clean empty folders
total_deleted_dirs = 0
for url_dir_name in tqdm(os.listdir(scrapped_htmls_dir), desc="Finding and deleting empty dirs"):
    url_dir = os.path.join( scrapped_htmls_dir, url_dir_name )
    files_count = len( os.listdir( url_dir ) )
    if files_count == 0:
        os.rmdir( url_dir )
        total_deleted_dirs += 1
print( "Total deleted empty dirs: {}".format( total_deleted_dirs ) )

site_images_scrapper = SiteHtmlScrapper()
#############
# debug
#uniq_url_hosts = ["-1", "#$155--0", "habr.com", "google.com"]
#uniq_url_hosts = uniq_url_hosts[:60]
#############

uniq_url_hosts = load( Path(interim_dir, "uniq_url_hosts.pkl") )
already_parsed_urls = []
for file in os.listdir(scrapped_htmls_dir):
    filename = os.fsdecode( file )
    already_parsed_urls.append( filename )
uniq_url_hosts = set(uniq_url_hosts).difference( already_parsed_urls )
uniq_url_hosts = list( uniq_url_hosts )

site_images_scrapper.scrap_sites_content(url_list = uniq_url_hosts, 
                                         save_dir = scrapped_htmls_dir, 
                                         max_scrolls = 4,
                                         scroll_pause = 0.5,
                                         n_jobs = 14)

    
print("done")