
import gc
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

class DataCleaner():
    def __init__(self):
        pass
    
    
    def unite_cathegories(self, df):
        
        df["cpe_manufacturer_name"].replace( "Huawei Device Company Limited", "Huawei", inplace=True )
        df["cpe_model_os_type"].replace( "Apple iOS", "iOS", inplace=True ) 
        
        gc.collect()
        
        return df
    
    def fill_missing_price(self, df):
        
        #df = df.copy()
        
        agg_group = "cpe_model_name"
        #agg_group = "cpe_manufacturer_name"
        
        all_prices = df["price"].values
        print("Calculating general nanmedian for filling price NaNs in special cases")
        global_median_price = np.nanmedian( all_prices )
        
        ###################################################
        # less correct but much faster than approach below
        df["price"] = df["price"].fillna( global_median_price )
        ###################################################
        
        """group_values = df[agg_group].values
        unique_group_values = np.unique(group_values)
        group_median_prices = {}
        #group_values_backward_index = self.build_backward_index_(group_values)
        for group_id in tqdm(unique_group_values, desc="Calculating median price for each {}".format(agg_group)):
            group_mask = group_values == group_id
            #group_mask = group_values_backward_index[ group_id ]
            group_prices = all_prices[ group_mask ]
            group_median_price = np.nanmedian( group_prices )
            group_median_prices[ group_id ] = group_median_price
            
        
        #model_price_vals = df[[agg_group, "price"]].values
        model_vals = df[agg_group].values
        for i in tqdm(range( len(all_prices) ), desc="Filling missing prices"):
            current_price = all_prices[i]
            if np.isnan( current_price ):
                current_model = model_vals[i]
                current_price = float( group_median_prices[current_model] )
                try:
                    if np.isnan( current_price ):
                        raise ValueError("Median price is nan for {}. Filling by global median {}".format(current_model, global_median_price))
                except ValueError as e:
                    current_price = global_median_price
                    print( e )
                all_prices[i] = current_price
        prices = all_prices.astype(np.float32)
        df["price"] = prices
        
        gc.collect()"""
        
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