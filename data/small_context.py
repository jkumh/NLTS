import darts.datasets
import pandas as pd
import numpy as np

dataset_names = [
    'AirPassengersDataset', 
    'AusBeerDataset', 
    'AustralianTourismDataset', 
    'ETTh1Dataset', 
    'ETTh2Dataset', 
    'ETTm1Dataset', 
    'ETTm2Dataset', 
    'ElectricityDataset', 
    'EnergyDataset', 
    'ExchangeRateDataset', 
    'GasRateCO2Dataset', 
    'HeartRateDataset', 
    'ILINetDataset', 
    'IceCreamHeaterDataset', 
    'MonthlyMilkDataset', 
    'MonthlyMilkIncompleteDataset', 
    'SunspotsDataset', 
    'TaylorDataset', 
    'TemperatureDataset',
    'TrafficDataset', 
    'USGasolineDataset', 
    'UberTLCDataset', 
    'WeatherDataset', 
    'WineDataset', 
    'WoolyDataset',
]

def get_descriptions(w_references=False):
    descriptions = []
    for dsname in dataset_names:
        d = getattr(darts.datasets,dsname)().__doc__
        
        if w_references:
            descriptions.append(d)
            continue

        lines = []
        for l in d.split("\n"):
            if l.strip().startswith("References"):
                break
            if l.strip().startswith("Source"):
                break
            if l.strip().startswith("Obtained"):
                break
            lines.append(l)
        
        d = " ".join([x.strip() for x in lines]).strip()

        descriptions.append(d)

    return dict(zip(dataset_names,descriptions))

def get_dataset(dsname):
    darts_ds = getattr(darts.datasets,dsname)().load()
    if dsname=='GasRateCO2Dataset':
        darts_ds = darts_ds[darts_ds.columns[1]]
    #series = darts_ds.pd_series()
    #series = darts_ds.pd_dataframe().iloc[:, 0]
    if hasattr(darts_ds, "pd_dataframe"):          # 0.24+
        series = darts_ds.pd_dataframe().iloc[:, 0]
    elif hasattr(darts_ds, "data"):                # 0.21-0.23
        series = darts_ds.data.iloc[:, 0]
    else:                                          
        series = pd.Series(darts_ds.values().flatten(),
                           index=darts_ds.time_index)
    if dsname == 'SunspotsDataset':
        series = series.iloc[::4]
    if dsname =='HeartRateDataset':
        series = series.iloc[::2]
    return series

def add_noise(series, noise_level=0.1):
    std_dev = np.std(series)  # 计算系列的标准差
    noise = np.random.normal(0, noise_level * std_dev, len(series))
    noisy_series = series + noise
    return noisy_series

def add_fixed_noise(series, noise_level=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed) 

    std_dev = np.std(series)
    noise = np.random.normal(0, noise_level * std_dev, len(series))
    noisy_series = series + noise
    return noisy_series

def get_datasets(n=-1,testfrac=0.2, noise=False, noise_level=0.01):
    datasets = [
        # 'AirPassengersDataset',
        # 'AusBeerDataset',
        # 'GasRateCO2Dataset', # multivariate
        'MonthlyMilkDataset',
        # 'SunspotsDataset', #very big, need to subsample?
        # 'WineDataset',
        # 'WoolyDataset',
        # 'HeartRateDataset',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        series = get_dataset(dsname)
        splitpoint = int(len(series)*(1-testfrac))
        
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]

        if noise:
            train = add_noise(train, noise_level)
        
        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def darts_spilt(n=-1, testfrac=0.2, valfrac=0.15,noise=False, noise_level=0.01):
    datasets = [
        'AirPassengersDataset',
        'AusBeerDataset',
        #'GasRateCO2Dataset', # multivariate
        #'MonthlyMilkDataset',
        'SunspotsDataset', #very big, need to subsample?
        'WineDataset',
        'WoolyDataset',
        #'HeartRateDataset',
    ]
    datas = {} 
    
    for i, dsname in enumerate(datasets):
        series = get_dataset(dsname)
        total_len = len(series)
        
        test_len = int(total_len * testfrac)
        val_len = int(total_len * valfrac)    
        
        splitpoint1 = total_len - test_len - val_len
        
        splitpoint2 = total_len - test_len
      
        # train: 0~splitpoint1-1
        train = series.iloc[:splitpoint1]
        # val: splitpoint1 ~ splitpoint2-1
        val = series.iloc[splitpoint1:splitpoint2]
        # test: splitpoint2 ~ end
        test = series.iloc[splitpoint2:]        
        if noise:
            train = add_noise(train, noise_level)
        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
  
    return datas

def get_memorization_datasets(n=-1,testfrac=0.15, predict_steps=30, noise=False, noise_level=0.01):
    datasets = [
        # 'IstanbulTraffic',
        # 'TSMCStock',
        'TurkeyPower'
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/memorization/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]

        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets, datas))
    
def memorization_split(n=-1, testfrac=0.15, valfrac=0.15, predict_steps=30, noise=False, noise_level=0.01):
    datasets = [
        'IstanbulTraffic',
        #'TSMCStock',
        #'TurkeyPower'
    ]
    datas = {} 
    for i, dsname in enumerate(datasets):
        try:
            with open(f"datasets/memorization/{dsname}.csv") as f:
                series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
                series = series.astype(float)
                series = pd.Series(series)
        except FileNotFoundError:
            print(f"Warning: can not find {dsname}.csv, please check the file path. Skipping this dataset.")
            continue
            
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            
            val_len = int(remaining_len * valfrac)
            
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            splitpoint1 = total_len - test_len - val_len
          
            splitpoint2 = total_len - test_len
       
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_informer_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01):
    datasets = [
        'ETTh1',
        'ETTm1',
        'ETTh2',
        'ETTm2',
        'electricity',
        'traffic',
        'weather'
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def informer_split(n=-1,testfrac=0.15, valfrac=0.15,predict_steps=96, noise=False, noise_level=0.01):
    datasets = [
        #'ETTh1',
        #'ETTm1',
        'ETTh2',
        #'ETTm2',
        #'electricity',
        #'traffic',
        #'weather'
    ]
    datas ={}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)

        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_national_illness_datasets(n=-1,testfrac=0.15, predict_steps=24, noise=False, noise_level=0.01):
    datasets = [
        'national_illness',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def national_illness_split(n=-1,testfrac=0.15,valfrac=0.15,predict_steps=24, noise=False, noise_level=0.01):
    datasets = [
        'national_illness',
    ]
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas
# different models
def get_dellm_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01):
    datasets = [
        'ETTh2',
        # 'traffic',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

# different number of samples
def get_num_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01):
    datasets = [
        'ETTh2',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

# different noise distributions
def add_def_noise(series, noise_level=0.1, noise_type='gaussian'):
    std_dev = np.std(series)  

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * std_dev, len(series))
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level * std_dev, noise_level * std_dev, len(series))
    elif noise_type == 'laplace':
        noise = np.random.laplace(0, noise_level * std_dev / np.sqrt(2), len(series))
    elif noise_type == 'beta':
        a, b = 2, 5 
        noise = np.random.beta(a, b, len(series)) * (2 * noise_level * std_dev) - noise_level * std_dev
    elif noise_type == 'geometric':
        p = 0.5  
        noise = np.random.geometric(p, len(series)) * noise_level * std_dev
    elif noise_type == 'gamma':
        shape = 2.0  
        scale = noise_level * std_dev  
        noise = np.random.gamma(shape, scale, len(series)) - np.mean(np.random.gamma(shape, scale, len(series)))  
    else:
        raise ValueError("not supported. Please choose from 'gaussian', 'uniform', 'laplace', 'beta', 'geometric', or 'gamma'")

    noisy_series = series + noise
    return noisy_series

def get_def_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01, noise_type='gaussian'):
    datasets = [
        'ETTh2',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level, noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def distribution_split(n=-1,testfrac=0.15,valfrac=0.15, predict_steps=96, noise=False, noise_level=0.01, noise_type='gaussian'):
    datasets = [
        'ETTh2',
    ]
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_def_noise(train, noise_level, noise_type)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_def2_datasets(n=-1,testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01, noise_type='gaussian'):
    datasets = [
        'ETTm1',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_def_noise(train, noise_level, noise_type)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def add_fixed_noise(series, noise_level=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed) 

    std_dev = np.std(series)
    noise = np.random.normal(0, noise_level * std_dev, len(series))
    noisy_series = series + noise
    return noisy_series, noise

def get_bv_datasets(n=-1, testfrac=0.15, predict_steps=96, noise=False, noise_level=0.01):
    datasets = [
        'traffic',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/ETT-small/processed/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train, noise_seq = add_fixed_noise(train, noise_level, seed=2025)
            datas.append((train, test, noise_seq)) 
        else:
            datas.append((train, test))           

        if i+1==n:
            break
    return dict(zip(datasets,datas))

def get_sy_datasets(n=-1,testfrac=0.15, predict_steps=30, noise=False, noise_level=0.1):
    datasets = [
        'ExpSineSquared_original',
        'Linear_original',
        'Matern_original',
        'Polynomial_original',
        'RBF_original',
        # 'RationalQuadratic_original',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/synthetic_dkernel2_1000/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]
        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def sy_split(n=-1,testfrac=0.15, valfrac=0.15,predict_steps=30, noise=False, noise_level=0.1):
    datasets = [
        'ExpSineSquared_original',
        'Linear_original',
        'Matern_original',
        'Polynomial_original',
        'RBF_original',
        # 'RationalQuadratic_original',
    ]
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/synthetic_dkernel2_1000/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_399007_datasets(n=-1,testfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    datasets = [
        '399007d',
        # '399007w',
        '399007m',
        '399007h',
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/399007/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]

        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def s399007_split(n=-1,testfrac=0.15,valfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    datasets = [
        '399007d',
        # '399007w',
        '399007m',
        '399007h',
    ]
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/399007/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_stock_datasets(n=-1,testfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    datasets = [
        # '000016d',
        # '000016h',
        # '000016m',
        # '000300d',
        # '000300h',
        # '000300m',
        # 'DJIAd',
        # 'DJIAh',
        # 'DJIAm',
        #'SPXh',
        'SPXm',        
    ]
    datas = []
    for i,dsname in enumerate(datasets):
        with open(f"datasets/stock/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]

        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def stock_split(n=-1,testfrac=0.15,valfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    datasets = [
        '000016d',
        '000016h',
        '000016m',
        '000300d',
        '000300h',
        '000300m',
        'DJIAd',
        'DJIAh',
        'DJIAm',
        'SPXh',
        'SPXm',        
    ]
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/stock/{dsname}.csv") as f:
            series = pd.read_csv(f, index_col=0, parse_dates=True).values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def get_sensor(type,axis,n=-1,testfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    datasets = [
        '3_drinking_2020_12_01_16_46_42',
    ]
    datas = []
    Racc  = 16384.0
    Rgyro = 16.4
    if type=='acc':
        r=Racc
    else:
        r=Rgyro
    for i,dsname in enumerate(datasets):
        with open(f"datasets/sensor/3_drinking/{dsname}.csv") as f:
            series = pd.read_csv(f,header=None).iloc[:,axis]/r
            series=series.values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        if predict_steps is not None:
            splitpoint = len(series)-predict_steps
        else:    
            splitpoint = int(len(series)*(1-testfrac))
        train = series.iloc[:splitpoint]
        test = series.iloc[splitpoint:]

        if noise:
            train = add_noise(train, noise_level)

        datas.append((train,test))
        if i+1==n:
            break
    return dict(zip(datasets,datas))

def sensor_split(type,axis,n=-1,testfrac=0.15,valfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    #axis=0:X,=1:Y,=2:Z
    #type=acc,gyro
    datasets = [
        '3_drinking_2020_12_01_16_46_42',
    ]
    Racc  = 16384.0
    Rgyro = 16.4
    if type=='acc':
        r=Racc
    else:
        r=Rgyro
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/sensor/3_drinking/{dsname}.csv") as f:
            series = pd.read_csv(f,header=None).iloc[:,axis]/r
            series=series.values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas

def cut_sensor_split(type,axis,n=-1,testfrac=0.15,valfrac=0.15, predict_steps=7, noise=False, noise_level=0.01):
    #axis=0:X,=1:Y,=2:Z
    #type=acc,gyro
    datasets = [
        '3_drinking_2020_12_01_16_46_42',
    ]
    Racc  = 16384.0
    Rgyro = 16.4
    if type=='acc':
        r=Racc
    else:
        r=Rgyro
    datas = {}
    for i,dsname in enumerate(datasets):
        with open(f"datasets/sensor/3_drinking/{dsname}.csv") as f:
            series = pd.read_csv(f,header=None).iloc[:,axis]/r
            series = series.tail(200).reset_index(drop=True)
            series=series.values.reshape(-1)
            # treat as float
            series = series.astype(float)
            series = pd.Series(series)
        total_len = len(series)
        if predict_steps is not None:
            test_len = predict_steps
            
            remaining_len = total_len - test_len
            val_len = int(remaining_len * valfrac)
            splitpoint1 = remaining_len - val_len
            splitpoint2 = remaining_len 
            
        else:
            test_len = int(total_len * testfrac)
            val_len = int(total_len * valfrac)
            
            
            splitpoint1 = total_len - test_len - val_len
            
            splitpoint2 = total_len - test_len
            
        train = series.iloc[:splitpoint1]
        val = series.iloc[splitpoint1:splitpoint2]
        test = series.iloc[splitpoint2:]

        if noise:
            train = add_noise(train, noise_level)

        datas[dsname] = (train, val, test)
        
        if i + 1 == n:
            break         
    return datas
