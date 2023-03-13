
import joblib
import pickle
import numpy as np
from tqdm import tqdm

def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as obj_file:
        pickle.dump( obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL )

    if verbose:
        print("Object saved to {}".format(path))
    pass

def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj

"""import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)"""

def build_backward_index(x_array):
            
    backward_index = {}
    for i in tqdm(range(len(x_array)), desc="Building backward index"):
        current_x = x_array[i]

        if current_x not in backward_index.keys():
            backward_index[current_x] = []
        backward_index[current_x].append(i)
        
    for x in tqdm(backward_index.keys(), desc="Building backward index (final types converting)"):
        backward_index[x] = np.array( backward_index[x] )
        
    return backward_index