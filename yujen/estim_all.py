import pandas as pd
import numpy as np
from epyestim import covid19 as cv19
from datetime import datetime
from functools import partial

cgrt = pd.read_csv('../data/OxCGRT_latest.csv', dtype={'RegionName':str, 'RegionCode': str}, 
                   parse_dates=['Date'], infer_datetime_format=True)

df = cgrt.drop(columns=['CountryCode', 'RegionCode', 'M1_Wildcard', 'E3_Fiscal measures', 'E4_International support',
    'H4_Emergency investment in healthcare', 'StringencyIndex', 'StringencyIndexForDisplay',
    'StringencyLegacyIndex', 'StringencyLegacyIndexForDisplay',
    'GovernmentResponseIndex', 'GovernmentResponseIndexForDisplay',
    'ContainmentHealthIndex', 'ContainmentHealthIndexForDisplay',
    'EconomicSupportIndex', 'EconomicSupportIndexForDisplay'])

def non_empty(df):
    b = pd.notnull(df['ConfirmedCases']).any()
    if not b:
        print('Filterd:', df.name)
    return b

country_region = ['CountryName', 'RegionName']
df = df.groupby(country_region, dropna=False).filter(non_empty)

def age_interval_transform(value):
    if pd.isnull(value):
        return 1
    elif '80+' in value:
        return 0.8
    else:
        return float(value.split('-',1)[0]) / 100

fillzero = ['C1_School closing',
       'C2_Workplace closing', 'C3_Cancel public events',
       'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements',
       'C7_Restrictions on internal movement', 'C8_International travel controls', 
       'E1_Income support', 'E2_Debt/contract relief', 
       'H1_Public information campaigns', 'H1_Flag', 'H2_Testing policy', 'H3_Contact tracing',
       'H5_Investment in vaccines',
       'H6_Facial Coverings', 'H7_Vaccination policy', 'H8_Protection of elderly people',
       'V1_Vaccine Prioritisation (summary)', 'V2A_Vaccine Availability (summary)',
       'V2D_Medically/ clinically vulnerable (Non-elderly)', 'V2E_Education',
       'V2F_Frontline workers  (non healthcare)', 'V2G_Frontline workers  (healthcare)',
       'V3_Vaccine Financial Support (summary)', 'V4_Mandatory Vaccination (summary)']
filln1 = ['C1_Flag', 'C2_Flag', 'C3_Flag', 'C4_Flag', 'C5_Flag', 'C6_Flag', 'C7_Flag', 'E1_Flag', 'H6_Flag',
          'H7_Flag', 'H8_Flag']
age_interval = ['V2B_Vaccine age eligibility/availability age floor (general population summary)',
       'V2C_Vaccine age eligibility/availability age floor (at risk summary)']
feat_cols = fillzero + filln1 + age_interval + ['Jurisdiction']

# fill null values
fill_dict = {k:0 for k in fillzero}
fill_dict.update({k:-1 for k in filln1})

df = df.fillna({'RegionName': '', 'RegionCode': ''})
df['Jurisdiction'] = df['Jurisdiction'].map({'NAT_TOTAL': 1, 'STATE_TOTAL': 2})
df[['ConfirmedCases', 'ConfirmedDeaths']] = (df.groupby(country_region, dropna=False)[['ConfirmedCases', 'ConfirmedDeaths']]
                                            .fillna(method='ffill').fillna(0))
df = df.fillna(value=fill_dict)
for c in age_interval:
    df[c] = df[c].apply(age_interval_transform)

# estimate R
def estim_R(df):
    print(df[['CountryName', 'RegionName']].iloc[0])
    df = df.copy()
    df.set_index('Date', inplace=True)
    cases = df['ConfirmedCases']
    R = cv19.r_covid(cases)
    return df.merge(R, how='inner', left_index=True, right_index=True)
    
def policy_change_delta_R(df, shift):
    print(df)
    print(df['CountryName'].unique())
    feats = df[feat_cols]
    R = df['R_mean']
    policy_change = (feats.iloc[shift+1:].values != feats.iloc[:-shift].values).any(axis=1)
    delta_R = R.iloc[shift+1:].values - R.iloc[:-shift].values
    return pd.DataFrame({'PolicyChange': policy_change, 'delta_R': delta_R})

estims = df.groupby(['CountryName', 'RegionName']).apply(estim_R)
temp = estims.reset_index(level=2).rename(columns={'level_2': 'Date'}).reset_index(drop=True)
temp.to_csv('../data/estims_all.csv')