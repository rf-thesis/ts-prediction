import pandas as pd

def getname(polygon):
    df_pinfo = pd.read_csv('data/2017_polygoninfo_wout1.csv', usecols=['ogr_fid', 'name'])
    df_pinfo = df_pinfo.set_index('ogr_fid')
    df_pinfo.index.rename('POLYGON', inplace=True)
    name = df_pinfo.loc[polygon].values
    return name[0]

def gettype(polygon):
    df_pinfo = pd.read_csv('data/2017_polygoninfo_wout1.csv', usecols=['ogr_fid', 'type'])
    df_pinfo = df_pinfo.set_index('ogr_fid')
    df_pinfo.index.rename('POLYGON', inplace=True)
    name = df_pinfo.loc[polygon].values
    return name[0]
