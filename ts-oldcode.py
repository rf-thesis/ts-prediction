# get polygon list
pols = df_orig.polygon_id.unique()
    pols = pd.DataFrame(pols)
    pols.to_csv('2017_devicecount15m-polygons.csv', index=False, header=False)