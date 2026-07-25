import pandas as pd


def add_time_features(df: pd.DataFrame) -> dict[str, pd.Series]:        #TO DO: change the session hours dinamically but not from the config file, from converting the index timezoe to the standard UTC

    new_cols = {}
    
    # 1. Componente de Timp de bază
    new_cols['hour'] = pd.Series(df.index.hour, index=df.index)
    new_cols['minute'] = pd.Series(df.index.minute, index=df.index)
    new_cols['day_of_week'] = pd.Series(df.index.dayofweek, index=df.index)
    new_cols['day_of_month'] = pd.Series(df.index.day, index=df.index)
    new_cols['month'] = pd.Series(df.index.month, index=df.index)
    
    # 2. Logica de Sesiuni (Vectorizată pentru viteză C)
    session_map = {}
    for h in range(24):
        if 0 <= h < 8:
            session_map[h] = 'asian'
        elif 8 <= h < 16:
            session_map[h] = 'london'
        elif 13 <= h < 21:
            session_map[h] = 'new_york'
        else:
            session_map[h] = 'off_hours'
            
    session_series = new_cols['hour'].map(session_map)
    new_cols['session'] = session_series
    
    # 3. One-hot encoding pentru Machine Learning (1 și 0)
    new_cols['is_asian'] = (session_series == 'asian').astype(int)
    new_cols['is_london'] = (session_series == 'london').astype(int)
    new_cols['is_new_york'] = (session_series == 'new_york').astype(int)
    
    return new_cols
