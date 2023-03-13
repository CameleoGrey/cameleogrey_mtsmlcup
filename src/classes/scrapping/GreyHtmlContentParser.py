
import os
import re
import numpy as np
import collections

from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy
from pprint import pprint

class GreyHtmlContentParser():
    def __init_(self):
        pass
    
    def parse_html_dir(self, html_dir, n_jobs=1):
        
        url_host_names = list( os.listdir( html_dir ) )
        # shuffle for a uniform performance on each thread
        np.random.seed(45)
        np.random.shuffle( url_host_names )
        ########
        # debug
        #url_host_names = url_host_names[:1000]
        #np.random.seed(45)
        #np.random.shuffle( url_host_names )
        ########
        url_host_names_batches = np.array_split( url_host_names, n_jobs )
        working_dir_batches = []
        for i in range(n_jobs):
            working_dir_batches.append( deepcopy(html_dir) )
        
        def parse_batch( url_host_names_batch, working_dir ):
            parsed_content_batch = []
            
            fill_empty_string = "***NO_CONTENT***"
            for url_host_name in tqdm(url_host_names_batch, desc="Parsing html files"):
                html_dir_path = Path( working_dir, url_host_name )
                html_files_count = len(os.listdir(html_dir_path))
                if html_files_count == 0:
                    parsed_content_batch.append( fill_empty_string )
                    continue
                
                html_file_path = Path( html_dir_path, "index.html" )
                html_text = read_html( html_file_path )
                
                parsed_tokens = parse_useful_content( html_text )
                if len(parsed_tokens) == 0:
                    parsed_content_batch.append( fill_empty_string )
                    continue
                
                #pprint(parsed_tokens)
                html_content = " ".join( parsed_tokens )
                html_content = html_content.strip()
                parsed_content_batch.append( html_content )
            
            return parsed_content_batch
        
        parsed_content = Parallel(n_jobs=n_jobs)( delayed(parse_batch)(uhnb, wd) for uhnb, wd in zip( url_host_names_batches, working_dir_batches ) )
        
        url_content_dict = {}
        for i in range(n_jobs):
            for j in range(len(parsed_content[i])):
                url_host = url_host_names_batches[i][j]
                html_content = parsed_content[i][j]
                url_content_dict[ url_host ] = html_content
        
        return url_content_dict
        
        
    
def read_html(html_path):
    with open( html_path, mode="r", encoding="utf-8" ) as html_file:
        html_text = html_file.readlines()
        html_text = "".join(html_text)
    return html_text

def parse_useful_content(html_text):
    
    html_text = clean_html_text( html_text )

    #parse <meta>
    meta_tokens = parse_meta_content_( html_text )
        
    #parse >text<
    inner_tokens = parse_inner_content_( html_text )
    
    parsed_tokens = meta_tokens + inner_tokens
    
    return parsed_tokens

def parse_inner_content_(html_text):
    parsed_tokens = []
    i = 0
    while i < len(html_text):
        current_char = html_text[i]
        if current_char == ">":
            i += 1
            if i >= len(html_text): break
            current_char = html_text[i]
            current_token = []
            while current_char != "<":
                current_token.append( current_char )
                i += 1
                current_char = html_text[i]
            if len( current_token ) > 0:
                current_token = "".join( current_token )
                parsed_tokens.append( current_token )
        i += 1
    return parsed_tokens
    

def parse_meta_content_(html_text):
    parsed_tokens = []
    meta_entry_ids = find_all_entries_(html_text, "<meta", overlapping = False)
    for i in range( len(meta_entry_ids) ):
        meta_start_pos = meta_entry_ids[i]
        meta_end_pos = meta_entry_ids[i]
        
        current_char = html_text[meta_end_pos]
        while current_char != ">":
            meta_end_pos += 1
            current_char = html_text[meta_end_pos]
        
        meta_string = html_text[meta_start_pos : meta_end_pos]
        content_entry_id = find_all_entries_(meta_string, "content=", overlapping = False)
        if len(content_entry_id) == 0:
            continue
        content_entry_id = content_entry_id[0]
        content_attribute_len = len("content=")
        content_entry_id += content_attribute_len
        current_token = []
        current_char = meta_string[ content_entry_id ]
        while current_char != ">":
            current_token.append( current_char )
            content_entry_id += 1
            if content_entry_id >= len(meta_string): break
            current_char = meta_string[ content_entry_id ]
        current_token = "".join( current_token )
        parsed_tokens.append( current_token )
    return parsed_tokens

def clean_html_text(html_text):
    html_text = re.sub("\n+", " ", html_text)
    html_text = re.sub("\t+", " ", html_text)
    html_text = re.sub("&nbsp", " ", html_text)
    html_text = re.sub(" +", " ", html_text)
    html_text = re.sub("> ", ">", html_text)
    html_text = re.sub(" <", "<", html_text)
    html_text = html_text.strip()
    
    bad_tag_starts = [ "<script", "<style", "<!--" ]
    bad_tag_ends = [ "/script>", "/style>", "-->" ]
    
    trash_entries_dict = {}
    for bad_tag_start, bad_tag_end in zip(bad_tag_starts, bad_tag_ends):
        
        tag_start_entries = find_all_entries_(html_text, bad_tag_start, overlapping = False)
        tag_end_entries = find_all_entries_(html_text, bad_tag_end, overlapping = False)
        
        for i in range(len(tag_end_entries)):
            tag_end_entries[i] += len(bad_tag_end)
        
        for i in range( len(tag_start_entries) ):
            try:
                trash_entries_dict[ tag_start_entries[i] ] = tag_end_entries[i]
            except Exception as e:
                if len(tag_start_entries) < len(tag_end_entries):
                    print("Didn't find pair end tag for {}".format(bad_tag_start))
                else:
                    print("Didn't find pair end tag for {}".format(bad_tag_end))
                print("Watch out for corrupted tokens in the parsed content!")
                
                continue
    trash_entries_dict = collections.OrderedDict(sorted(trash_entries_dict.items()))
    
    entries_to_remove = []
    for entry_id in trash_entries_dict.keys():
        entry_start = entry_id
        entry_end = trash_entries_dict[entry_id]
        if entry_start >= entry_end:
            entries_to_remove.append( entry_start )
    for entry_to_remove in entries_to_remove:
        del trash_entries_dict[entry_to_remove]
    
    trash_mask_char = "\x25A0" # black square
    pure_html_text = []
    i = 0
    while i < len( html_text ):
        if i in trash_entries_dict.keys():
            trash_end_id = trash_entries_dict[i]
            while i != trash_end_id:
                pure_html_text.append( trash_mask_char )
                i += 1
        else:
            pure_html_text.append( html_text[i] )
            i += 1
    pure_html_text = "".join( pure_html_text )
    pure_html_text = re.sub("{}+".format(trash_mask_char), "", pure_html_text)
        
    return pure_html_text

def find_all_entries_( source_string, substring, overlapping = False):
    
    entry_ids = []
    string_len = len(source_string)
    i = 0
    while True:
        i = source_string.find(substring, i)
        
        if i == -1:
            break
        
        entry_ids.append( i )
        
        if overlapping:
            i += 1
        else:
            i += len(substring)
    
    return entry_ids
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    