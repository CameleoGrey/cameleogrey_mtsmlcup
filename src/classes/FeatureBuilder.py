
import gc
import sys
import math
import numpy as np
import pandas as pd
import calendar

from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from classes.GreyCategoricalEncoder import GreyCategoricalEncoder

class FeatureBuilder():
    def __init__(self):
        
        self.cat_encoders = {}
        self.url_encoder = None
        
        self.cat_encoders_fitted = False
        self.url_encoder_fitted = False
        
        pass
    
    
    def collect_count_aggregates(self, df):
        
        print("Collecting count aggregates")
        
        user_count_agg_feats = {}
        
        user_ids = df["user_id"].values
        day_names = df["day_name"].values
        part_of_days = df["part_of_day"].values
        day_name_parts = df["day_name_part"].values
        request_counts = df["request_cnt"].values
        
        uniq_user_ids = np.unique( user_ids )
        # numpy has troubles with string arrays, so will use set() and etc...
        uniq_day_names = np.array( list( sorted(list( set(day_names) ) ) ) )
        uniq_part_of_days = np.array( list( sorted(list( set(part_of_days) ) ) ) )
        uniq_day_name_parts = np.array( list( sorted(list( set(day_name_parts) ) ) ) )
        
        count_agg_feat_names = []
        for day_name in uniq_day_names:
                count_agg_feat_names.append( "abs request_cnt by " + day_name )
                count_agg_feat_names.append( "rel request_cnt by " + day_name )
        
        for part_of_day in uniq_part_of_days:
                count_agg_feat_names.append( "abs request_cnt by " + part_of_day )
                count_agg_feat_names.append( "rel request_cnt by " + part_of_day )
        
        for day_name_part in uniq_day_name_parts:
                count_agg_feat_names.append( "abs request_cnt by " + day_name_part )
                count_agg_feat_names.append( "rel request_cnt by " + day_name_part )
        count_agg_feat_names.append( "abs sum_requests" )
        count_agg_feat_names.append( "rel sum_requests" )
                      
        user_count_agg_feats["feature_names"] = count_agg_feat_names
        
        user_ids_backward_index = self.build_backward_index_( user_ids )
        
        each_user_requests_sum_counts = []
        for current_user_id in tqdm(uniq_user_ids):
            #user_mask = user_ids == current_user_id
            user_mask = user_ids_backward_index[ current_user_id ]
            user_request_counts = request_counts[ user_mask ]
            user_requests_sum = np.sum( user_request_counts )
            each_user_requests_sum_counts.append( user_requests_sum )
        general_median_user_requests = np.median( each_user_requests_sum_counts )
        
        for current_user_id in tqdm(uniq_user_ids, desc="Building count aggregates for each user_id"):
            #user_mask = user_ids == current_user_id
            user_mask = user_ids_backward_index[ current_user_id ]
            user_request_counts = request_counts[ user_mask ]
            user_day_names = day_names[ user_mask ]
            user_part_of_days = part_of_days[ user_mask ]
            user_day_name_parts = day_name_parts[ user_mask ]
            user_requests_sum = np.sum( user_request_counts )
            
            current_user_count_feats = []
            
            for uniq_day_name in uniq_day_names:
                user_day_name_mask = user_day_names == uniq_day_name
                user_day_name_counts = user_request_counts[ user_day_name_mask ]
                user_day_name_request_sum = np.sum( user_day_name_counts )
                current_user_count_feats.append( user_day_name_request_sum )
                current_user_count_feats.append( user_day_name_request_sum / user_requests_sum )
            
            for uniq_part_of_day in uniq_part_of_days:
                user_part_of_day_mask = user_part_of_days == uniq_part_of_day
                user_part_of_day_counts = user_request_counts[ user_part_of_day_mask ]
                user_part_of_day_request_sum = np.sum( user_part_of_day_counts )
                current_user_count_feats.append( user_part_of_day_request_sum )
                current_user_count_feats.append( user_part_of_day_request_sum / user_requests_sum )
            
            for uniq_day_name_part in uniq_day_name_parts:
                user_day_name_part_mask = user_day_name_parts == uniq_day_name_part
                user_day_name_part_counts = user_request_counts[ user_day_name_part_mask ]
                user_day_name_part_request_sum = np.sum( user_day_name_part_counts )
                current_user_count_feats.append( user_day_name_part_request_sum )
                current_user_count_feats.append( user_day_name_part_request_sum / user_requests_sum )
            
            current_user_count_feats.append( user_requests_sum )
            current_user_count_feats.append( user_requests_sum / general_median_user_requests )
            
            user_count_agg_feats[current_user_id] = np.array( current_user_count_feats )
        
        gc.collect()
            
        return user_count_agg_feats
    
    def collect_encoded_cat_features_aggregates(self, df, base_cat_feature_names):
        
        # aggregate by user id
        # base cat_feats names: 
        #"region_name", "city_name", "cpe_manufacturer_name", 
        #"cpe_model_name", "cpe_type_cd", "cpe_model_os_type",
        #"part_of_day", "day_name", "day_name_part"
        # aggregates: min, max, mean, median, std
        
        
        # extract column names of encoded cat features
        feature_columns = list(df.columns)
        corresponced_columns = {}
        for i in range(len(base_cat_feature_names)):
            current_base_name = base_cat_feature_names[i]
            for j in range(len(feature_columns)):
                current_column_name = feature_columns[j]
                clean_col_name = current_column_name.split("_")
                clean_col_name = "_".join(clean_col_name[:len(clean_col_name)-1])
                if current_base_name == clean_col_name:
                    if current_base_name not in corresponced_columns.keys():
                        corresponced_columns[current_base_name] = []
                    corresponced_columns[current_base_name].append( current_column_name )
        
        
        applied_aggregates = ["min", "max", "mean", "median", "std"]
        user_cat_agg_feats = {}
        
        # build names for features for the future dataset
        cat_agg_feat_names = []
        for base_cat_name in base_cat_feature_names:
            for encoded_cat_col in corresponced_columns[base_cat_name]:
                for agg_name in applied_aggregates:
                    agg_feat_name = encoded_cat_col + "_{}".format( agg_name )
                    cat_agg_feat_names.append( agg_feat_name )
        user_cat_agg_feats["feature_names"] = cat_agg_feat_names
        
        
        user_ids = df["user_id"].values
        uniq_user_ids = np.unique( user_ids )
        user_ids_backward_index = self.build_backward_index_( user_ids )
        for current_user_id in tqdm(uniq_user_ids, desc="Building aggregates for categorical features for each user_id"):
            #user_mask = user_ids == current_user_id
            user_mask = user_ids_backward_index[ current_user_id ]
            current_user_cat_feats = []
            
            for base_cat_name in base_cat_feature_names:
                for encoded_cat_col in corresponced_columns[base_cat_name]:
                    cat_feat_values = df[encoded_cat_col].values
                    selected_user_cat_feat = cat_feat_values[user_mask]
                    
                    if "min" in applied_aggregates:
                        current_user_cat_feats.append( np.min(selected_user_cat_feat) )
                    if "max" in applied_aggregates:
                        current_user_cat_feats.append( np.max(selected_user_cat_feat) )
                    if "mean" in applied_aggregates:
                        current_user_cat_feats.append( np.mean(selected_user_cat_feat) )
                    if "median" in applied_aggregates:
                        current_user_cat_feats.append( np.median(selected_user_cat_feat) )
                    if "std" in applied_aggregates:
                        current_user_cat_feats.append( np.std(selected_user_cat_feat) )
            
            current_user_cat_feats = np.array( current_user_cat_feats )
            user_cat_agg_feats[current_user_id] = current_user_cat_feats
        
        gc.collect()
                    
        return user_cat_agg_feats
    
    def build_url_frequency_dict(self, df):
        
        url_hosts = df["url_host"].values
        user_ids = df["user_id"].values
        request_counts = df["request_cnt"].values
        
        url_frequency_dict = {}
        url_hosts_backward_index = self.build_backward_index_( url_hosts )
        uniq_url_hosts = np.unique( list( set( url_hosts ) ) )
        for uniq_url_host in tqdm( uniq_url_hosts, desc="Extracting url counts for each user"):
            url_frequency_dict[uniq_url_host] = {}
            url_mask = url_hosts_backward_index[ uniq_url_host ]
            current_url_users = user_ids[ url_mask ]
            current_url_counts = request_counts[ url_mask ]
            
            #url_frequency_dict[uniq_url_host]["overall_count"] = np.sum( current_url_counts )
            
            for i in range(len(current_url_users)):
                current_user_id = current_url_users[i]
                current_request_count = current_url_counts[i]
                
                if current_user_id not in url_frequency_dict[uniq_url_host].keys():
                    url_frequency_dict[uniq_url_host][current_user_id] = 0
                
                url_frequency_dict[uniq_url_host][current_user_id] += current_request_count
        
        return url_frequency_dict
    
    def fit_cat_encoders(self, df, feature_names, 
                         embedding_sizes, shuffle_counts):
        
        for i in tqdm(range(len(feature_names)), desc="Fitting category encoders"):
            feature_list = df[feature_names[i]].to_list()
            self.cat_encoders[feature_names[i]] = GreyCategoricalEncoder( shuffle_count=shuffle_counts[i], vec_size=embedding_sizes[i] )
            self.cat_encoders[feature_names[i]].fit( feature_list, feature_names[i] )
        
        self.cat_encoders_fitted = True
        
        gc.collect()
        
        return self
    
    def transform_cat_features(self, df):
        
        if self.cat_encoders_fitted is False:
            raise Exception("Category encoders are not fitted.")
        
        #df = df.copy()
        
        for feature_name in tqdm(self.cat_encoders.keys(), desc="Encoding categorical features"):
            feature_values = df[feature_name].to_list()
            embeddings, column_names = self.cat_encoders[feature_name].transform( feature_values )
            del df[feature_name]
            for i in tqdm(range(len(column_names)), desc="Replacing original \"{}\" by encoded".format(feature_name)):
                if len(column_names) == 1:
                    df[column_names[i]] = embeddings[:]
                else:
                    df[column_names[i]] = embeddings[:, i]
        
        gc.collect()
        
        return df
    
    def collect_price_aggregates(self, df):
        
        user_price_agg = {}
        user_price_agg["feature_names"] = []
        
        applied_aggregates = ["min", "max", "mean", "median", "std"]
        for app_agg in applied_aggregates:
            user_price_agg["feature_names"].append( "price_{}".format(app_agg) )
        
        user_ids = df["user_id"].values
        prices = df["price"].values
        uniq_user_ids = np.unique( user_ids )
        
        user_ids_backward_index = self.build_backward_index_( user_ids )
                
        for current_user_id in tqdm(uniq_user_ids, desc="Building price aggregates for each user_id"):

            #user_mask = user_ids == current_user_id
            user_mask = user_ids_backward_index[current_user_id]
            user_prices = prices[user_mask]
            
            current_user_price_feats = []
                
            if "min" in applied_aggregates:
                current_user_price_feats.append( np.min(user_prices) )
            if "max" in applied_aggregates:
                current_user_price_feats.append( np.max(user_prices) )
            if "mean" in applied_aggregates:
                current_user_price_feats.append( np.mean(user_prices) )
            if "median" in applied_aggregates:
                current_user_price_feats.append( np.median(user_prices) )
            if "std" in applied_aggregates:
                current_user_price_feats.append( np.std(user_prices) )
            
            current_user_price_feats = np.array( current_user_price_feats )
            user_price_agg[current_user_id] = current_user_price_feats
        
        
        return user_price_agg
        
    
    def log_price(self, df):
        #df = df.copy()
        
        prices = df["price"].values
        prices = np.log1p( prices )
        scaler = StandardScaler()
        scaler.fit( prices.reshape(-1, 1) )
        prices = scaler.transform( prices.reshape(-1, 1) )
        prices = prices.reshape(-1, )
        
        df["price"] = prices
        
        gc.collect()
        
        return df
    
    def add_day_names(self, df):
        
        #df = df.copy()
        
        dates = df["date"].values
        day_names = []
        
        uniq_dates = np.unique( dates )
        dates_day_names_dict = {}
        for uniq_date in uniq_dates:
            current_date = datetime.utcfromtimestamp( uniq_date.tolist()/1e9 )
            
            day_id = current_date.weekday()
            current_day_name = calendar.day_name[ day_id ]
            current_day_name = current_day_name[:2]
            current_day_name = current_day_name.lower()
            
            # skip full day name for reducing memory consumption
            #current_day_name = str(current_date.weekday())
            
            dates_day_names_dict[ uniq_date ] = current_day_name
            
        
        for i in tqdm(range(len(dates)), desc="Building day names"):
            current_day_name = dates_day_names_dict[ dates[i] ]
            day_names.append( current_day_name )
        
        df["day_name"] = day_names
        
        gc.collect()
        
        return df
    
    def crop_part_of_day(self, df):
        
        part_of_days = df["part_of_day"].values
        
        short_variants = {}
        short_variants["morning"] = "m"
        short_variants["day"] = "d"
        short_variants["evening"] = "e"
        short_variants["night"] = "n"
        
        short_parts = []
        for i in tqdm( range(len(part_of_days)), desc="Cropping part of days" ):
            current_pod = short_variants[ part_of_days[i] ]
            short_parts.append( current_pod )
        df["part_of_day"] = short_parts
        
        gc.collect()
        
        return df
        
    
    def add_day_name_part_feature(self, df):
        
        #df = df.copy()
        
        # much slower than second version but much more memory efficient
        day_names = df["day_name"].values
        day_parts = df["part_of_day"].values
        day_name_parts = []
        for i in tqdm( range(len(day_names)), desc="Building day_name + part_of_day feature" ):
            current_day_name_part = day_names[i] + " " + day_parts[i]
            day_name_parts.append( current_day_name_part )
        df["day_name_part"] = day_name_parts
        
        
        #start_time = datetime.now()
        #print("Combining day_name + part_of_day")
        #df["day_name_part"] = df["day_name"] + df["part_of_day"].astype(str)
        #total_time = datetime.now() - start_time
        #print("Combine time: {}".format( total_time ))
        
        #print(df["day_name_part"].values[:100])
        
        gc.collect()
        
        return df
    
    def build_backward_index_(self, x_array):
            
        backward_index = {}
        for i in tqdm(range(len(x_array)), desc="Building backward index"):
            current_x = x_array[i]

            if current_x not in backward_index.keys():
                backward_index[current_x] = []
            backward_index[current_x].append(i)
            
        for x in tqdm(backward_index.keys(), desc="Building backward index (final types converting)"):
            backward_index[x] = np.array( backward_index[x] )
            
        
        return backward_index
    
    
    