
import os
import gc
import time
import numpy as np
import asyncio

from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

class SiteHtmlScrapper():
    def __init__(self):
        pass
    
    def scrap_sites_content(self, url_list, save_dir, max_scrolls=4, scroll_pause=0.5, n_jobs=1):
        
        if n_jobs == 1:
            self.scrap_single_process_(url_list, save_dir, max_scrolls, scroll_pause)
        else:
            self.scrap_multicore_(url_list, save_dir, scroll_pause, n_jobs)
        pass
    
    def scrap_multicore_(self, url_list, save_dir, scroll_pause=0.5, n_jobs=2):
        
        def scrap_batch(url_batch, save_dir, scroll_pause):
            options = webdriver.ChromeOptions()
            #options.add_argument("--start-maximized")
            prefs = {'profile.default_content_setting_values': {'cookies': 2, 'images': 2, 'javascript': 2, 
                            'plugins': 2, 'popups': 2, 'geolocation': 2, 
                            'notifications': 2, 'auto_select_certificate': 2, 'fullscreen': 2, 
                            'mouselock': 2, 'mixed_script': 2, 'media_stream': 2, 
                            'media_stream_mic': 2, 'media_stream_camera': 2, 'protocol_handlers': 2, 
                            'ppapi_broker': 2, 'automatic_downloads': 2, 'midi_sysex': 2, 
                            'push_messaging': 2, 'ssl_cert_decisions': 2, 'metro_switch_to_desktop': 2, 
                            'protected_media_identifier': 2, 'app_banner': 2, 'site_engagement': 2, 
                            'durable_storage': 2}}
            options.add_experimental_option('prefs', prefs)
            options.add_argument("--incognito")
            options.add_argument("--disable-javascript")
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument("disable-infobars")
            options.add_argument("--disable-extensions")
            browser = webdriver.Chrome(options=options)
            browser.set_script_timeout(5)
            browser.set_page_load_timeout(5)
            
            for i in tqdm(range(len(url_batch)), desc="Scrapping site images"):
                
                current_url_host = url_batch[i]
                
                current_site_images_dir = Path( save_dir, current_url_host )
                if not Path( current_site_images_dir ).exists():
                    Path( current_site_images_dir ).mkdir(parents=True, exist_ok=True)
                else:
                    # skip already scrapped sites
                    continue
                
                if "ua" in current_url_host:
                    continue
                
                # just create empty directory
                if "." not in current_url_host:
                    continue
                    
                full_url = "http://{}".format( current_url_host )
                
                try:
                    browser.get( full_url )
                    #time.sleep(1)
                except Exception as e:
                    print("loading timeout")
    
                # simple version
                try:
                    with open(os.path.join(current_site_images_dir, "index.html".format(0)), "w", encoding='utf-8') as f:
                        f.write(browser.page_source)
                    """browser.save_screenshot(Path(current_site_images_dir, "{}.png".format(0)))
                    action_chain  = ActionChains(browser)
                    action_chain.send_keys(Keys.PAGE_DOWN).perform()
                    time.sleep( scroll_pause )
                    browser.save_screenshot(Path(current_site_images_dir, "{}.png".format(1)))
                    time.sleep( scroll_pause )"""
                except Exception as e:
                    print("Problem with browser occured, relaunching")
                    # build fresh instance of browser
                    # if it will fail, then internet connection problems --> stop working
                    del browser
                    gc.collect()
                    browser = webdriver.Chrome(options=options)
                    browser.set_script_timeout(5)
                    browser.set_page_load_timeout(5)
                    
                    try:
                        browser.get( full_url )
                        #time.sleep(1)
                    except Exception as e:
                        print("loading timeout")
                    try:
                        with open(os.path.join(current_site_images_dir, "index.html".format(0)), "w", encoding='utf-8') as f:
                            f.write(browser.page_source)
                    except Exception as e:
                        print("Failed to continue after browser relaunching. ")
                        return 
                    
                    
            
        
        url_batches = np.array_split( url_list, n_jobs )
        Parallel(n_jobs=n_jobs)(delayed(scrap_batch)(url_batch, save_dir, scroll_pause) for url_batch in url_batches)
        
        pass
    
    def scrap_single_process_(self, url_list, save_dir, max_scrolls=4, scroll_pause=0.5):
        
        options = webdriver.ChromeOptions()
        #options.add_argument("--start-maximized")
        prefs = {'profile.default_content_setting_values': {'cookies': 2, 'images': 2, 'javascript': 2, 
                            'plugins': 2, 'popups': 2, 'geolocation': 2, 
                            'notifications': 2, 'auto_select_certificate': 2, 'fullscreen': 2, 
                            'mouselock': 2, 'mixed_script': 2, 'media_stream': 2, 
                            'media_stream_mic': 2, 'media_stream_camera': 2, 'protocol_handlers': 2, 
                            'ppapi_broker': 2, 'automatic_downloads': 2, 'midi_sysex': 2, 
                            'push_messaging': 2, 'ssl_cert_decisions': 2, 'metro_switch_to_desktop': 2, 
                            'protected_media_identifier': 2, 'app_banner': 2, 'site_engagement': 2, 
                            'durable_storage': 2}}
        options.add_experimental_option('prefs', prefs)
        options.add_argument("--incognito")
        options.add_argument("--disable-javascript")
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        browser = webdriver.Chrome(options=options)
        
        # if connection timeout is too small then
        # duplicated images will be created for already another url
        browser.set_script_timeout(5)
        browser.set_page_load_timeout(5)
        
        for i in tqdm(range(len(url_list)), desc="Scrapping site images"):
            current_url_host = url_list[i]
            
            current_site_images_dir = Path( save_dir, current_url_host )
            if not Path( current_site_images_dir ).exists():
                Path( current_site_images_dir ).mkdir(parents=True, exist_ok=True)
            else:
                # skip already scrapped sites
                continue
            
            # just create empty directory
            if "." not in current_url_host:
                continue
                
            full_url = "http://{}".format( current_url_host )
            
            try:
                browser.get( full_url )
            except Exception as e:
                print("loading timeout")
            
            # simple version
            try:
                with open(os.path.join(current_site_images_dir, "index.html".format(0)), "w", encoding='utf-8') as f:
                        f.write(browser.page_source)
                """browser.save_screenshot(Path(current_site_images_dir, "{}.png".format(0)))
                action_chain  = ActionChains(browser)
                action_chain.send_keys(Keys.PAGE_DOWN).perform()
                time.sleep( scroll_pause )
                browser.save_screenshot(Path(current_site_images_dir, "{}.png".format(1)))
                time.sleep( scroll_pause )"""
            except Exception as e:
                print(e)
            
            """for j in range( max_scrolls ):
                
                browser.save_screenshot(Path(current_site_images_dir, "{}.png".format(j)))
                
                #last_height = browser.execute_script("return document.body.scrollHeight")
                last_offset = browser.execute_script("return document.body.scrollY ")
                html.send_keys(Keys.PAGE_DOWN)
                #browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep( scroll_pause )
                
                new_offset = browser.execute_script("return document.body.scrollY")
                if new_offset == last_offset:
                    break
                if j >= max_scrolls:
                    break
                #last_offset = new_offset
                
                #new_height = browser.execute_script("return document.body.scrollHeight")
                #if new_height == last_height:
                #    break
                #last_height = new_height"""
        
        pass
    