startdate = dateutil.parser.parse('2017-06-26 12:00:00')    # goes from 26-06 to 05-07
enddate =   dateutil.parser.parse('2017-06-30 12:00:00')

filename_data = '2017_devicecount15m.csv'
filename_pols = '2017_devicecount15m-polygons.csv'
basepath = 'data/'

# load data
df_orig = pd.read_csv(basepath + filename_data)
df_polygons = pd.read_csv(basepath + filename_pols, nrows=10)  # TODO:
list_polygons = df_polygons.values.flatten()

