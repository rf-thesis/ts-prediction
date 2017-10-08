# get polygon list
pols = df_orig.polygon_id.unique()
    pols = pd.DataFrame(pols)
    pols.to_csv('count_devices15m_2017_polygonlist.csv', index=False, header=False)