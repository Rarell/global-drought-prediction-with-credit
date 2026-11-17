

INPUT_VARIABLE_NAMES = [
    'u_200mb',
    'u_500mb',
    'v_200mb',
    'v_500mb',
    'z_200mb',
    'z_500mb',
    'q_tot_200mb',
    'q_tot_500mb',
    'tair',
    'sp',
    'tp',
    'd2m',
    'e',
    'pev',
    'ssr',
    'ws',
    'fg10',
    'enso',
    'iod',
    'SSR',
    'lsm',
    'tvh',
    'tvl',
    'cvh',
    'cvl',
]

INPUT_VARIABLE_SYMBOLIC_NAMES = [
    r'$U_{200mb}$',
    r'$U_{500mb}$',
    r'$V_{200mb}$',
    r'$V_{500mb}$',
    r'$Z_{200mb}$',
    r'$Z_{500mb}$',
    r'$q_{tot,200mb}$',
    r'$q_{tot}$', # Removing 500mb for now due to size
    r'$T$',
    r'$P_{surf}$',
    r'$Prec_{1day}$',
    r'$T_{dew}$',
    r'$ET$',
    r'$PET$',
    r'$R_{s}$',
    r'$WS$',
    r'$Gust$',
    r'$ENSO$',
    r'$IOD$',
    r'$R_{s,clim}$',
    r'Land-Sea Mask',
    r'Type of High Veg.',
    r'Type of Low Veg.',
    r'High Veg. Coverage',
    r'Low Veg. Coverage',
]

OUTPUT_VARIABLE_NAMES = [
    'u_200mb', # 200 mb; 0
    'u_500mb', # 500 mb; 1
    'v_200mb', # 200 mb; 2
    'v_500mb', # 500 mb; 3
    'z_200mb', # 200 mb; 4
    'z_500mb', # 500 mb; 5
    'q_tot_200mb', # 200 mb; 6
    'q_tot_500mb', # 500 mb; 7
    'tair', # 8
    'sp', # 9
    'tp', # 10
    'd2m', # 11
    'e', # 12
    'pev', # 13
    'tp_30d', # 14
    'swvl1', # 15
    'swvl2', # 16
    'swvl3', # 17
    'swvl4', # 18
    'ndvi', # 19
    'evi', # 20
    'lai', # 21
    'fpar', # 22
]

OUTPUT_VARIABLE_SYMBOLIC_NAMES = [
    r'$U_{200mb}$',
    r'$U_{500mb}$',
    r'$V_{200mb}$',
    r'$V_{500mb}$',
    r'$Z_{200mb}$',
    r'$Z_{500mb}$',
    r'$q_{tot,200mb}$',
    r'$q_{tot,500mb}$',
    r'$T$',
    r'$P_{surf}$',
    r'$Prec_{1day}$',
    r'$T_{dew}$',
    r'$ET$',
    r'$PET$',
    r'$Prec_{30day}$',
    r'$SM_{0-10cm}$',
    r'$SM_{10-40cm}$',
    r'$SM_{40-100cm}$',
    r'$SM_{100-200cm}$',
    r'$NDVI$',
    r'$EVI$',
    r'$LAI$',
    r'$fPAR$',
]
