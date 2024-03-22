from numba.pycc import CC
import numpy as np
import pandas as pd
import time

def distance(lon1, lat1, lon2, lat2):
    '''                                                                         
    Calculate the circle distance between two points                            
    on the earth (specified in decimal degrees)                                 
    '''
    # convert decimal degrees to radians                                        
    lon1, lat1 = map(np.radians, [lon1, lat1])
    lon2, lat2 = map(np.radians, [lon2, lat2])

    # haversine formula                                                         
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # 6367 km is the radius of the Earth                                        
    km = 6367 * c
    m = km * 1000
    return m

if __name__ == '__main__':

    # Use Numba to compile this same function in a module named `aot`
    # in both a vectorized and non-vectorized form
    cc = CC('aot')

    @cc.export('distance', 'f8(f8,f8,f8,f8)')
    @cc.export('distance_v', 'f8[:](f8[:],f8[:],f8,f8)')
    def distance_numba(lon1, lat1, lon2, lat2):
        '''                                                                         
        Calculate the circle distance between two points                            
        on the earth (specified in decimal degrees)
    
        (distance: Numba-accelerated; distance_v: Numba-accelerated + vectorized)
        '''
        # convert decimal degrees to radians                        
        lon1, lat1 = map(np.radians, [lon1, lat1])
        lon2, lat2 = map(np.radians, [lon2, lat2])

        # haversine formula                                                         
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # 6367 km is the radius of the Earth                                        
        km = 6367 * c
        m = km * 1000
        return m

    cc.compile()

    import aot # import in module we just compiled

    # load in AirBnB listings + coordinates of MACSS Building
    df = pd.read_csv('listings_chi.csv')
    macss = {'longitude': -87.5970978, 'latitude': 41.7856443}
    
    # time NumPy implementation
    t0 = time.time()
    df.loc[:,'distance_from_macss'] = distance(df.longitude, 
                                               df.latitude,                  
                                               macss['longitude'],
                                               macss['latitude'])
    t1 = time.time()
    
    # time Numba pre-compiled (vectorized) solution
    df.loc[:,'distance_from_macss'] = aot.distance_v(df.longitude.values, 
                                                     df.latitude.values,                  
                                                     macss['longitude'],
                                                     macss['latitude'])
    t2 = time.time()

    print(f'NumPy:{t1 - t0}, Numba (vectorized): {t2 - t1}')
