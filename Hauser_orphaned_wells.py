#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:35:20 2024

@author: gracehauser
"""

#%%

# =============================================================================
# Set-up 
# =============================================================================

# Title: "Hauser_orphaned_wells.py"
# Author: Grace Hauser
# Affiliations: YSPH Department of EHS & FracTracker Alliance Western Division
# Date of last update: 11/11/2024
# Script aim: answer the following questions
#             (1) How many orphaned wells are there in 2024?
#             (2) How many wells have become orphaned since the USGS report? 
#             (3) How many wells have been plugged since the USGS report?

# USGS publication: https://pubs.usgs.gov/publication/dr1167/full
# USGS dataset: https://www.sciencebase.gov/catalog/item/62ebd67bd34eacf539724c56
# USGS methodology: https://www.sciencebase.gov/catalog/file/get/62ebd67bd34eacf539724c56?f=__disk__fc%2F1e%2Fc2%2Ffc1ec2c6bd83535801cbaea9e17cf4dbf091a946&transform=1&allowOpen=true
# Fractracker dataset: available upon request at https://www.fractracker.org/data/

# Set working directory
import os
os.chdir('/Users/gracehauser/Desktop/FRACTRACKER/INDEPENDENT_PROJECT/DATASETS/WELLS')

# Load packages
import numpy as np
import pandas as pd
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer

#%%

# =============================================================================
# =============================================================================
# =============================================================================
# AIM 1 : HOW MANY ORPHANED WELLS ARE THERE IN 2024?
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Set-up: orphaned & plugged dictionaries for FT
# =============================================================================

# Orphaned dictionary 
state_status_dict = { 
    'Alabama' : ['Abandoned'],
    # Alaska : no category
    'Arkansas' : ['Abandoned Orphaned Well'],
    # California : no category
    # Colorado : no category
    # Florida : no category
    'Indiana' : ['Orphaned'],
    'Kansas': ['D&A'],
    'Kentucky' : ['AB',
                  'D&A',
                  'ABD'],
    'Louisiana' : ['23',
                   '26'],
    'Michigan' : ['Orphan'],
    'Mississippi' : ['PO - Potential Orphan Well',
                     'O - Orphaned Well'],
    'Missouri' : ['Abandoned',
                  'Abandoned, Unknown Location',
                  'Abandoned, Known Location and Verified',
                  'Abandoned, No evidence of existence/ Unable to find',
                  'Orphaned'],
    # Montana : no category
    'Nebraska' : ['AB', 'SI'],
    'Nevada' : ['AB'],
    'New Mexico' : ['Reclamation Fund Approved'],
    'New York' : ['UN',
                  'UL',
                  'UM'],
    'North Dakota' : ['AB'],
    'Ohio' : ['OR',
              'OP'],
    'Oklahoma' : ['OR'],
    'Pennsylvania' : ['DEP Orphan List'],
    'South Dakota' : ['Abandoned-Not Regulated']
    # Tennessee : no category
    # Texas : no category
    # Utah : no category
    # West Virginia : no category
    # Wyoming : no category
}

# Plugged dictionary
plugged_dict = { 
    'Alabama' : ['Plugged and Abandoned',  
                'Plugged Back'],
    'Alaska' : ['Plugged & Abandoned',
              'Surface Plug'],
    'Arkansas' : ['Plugged and Abandoned'],
    'California' : ['Plugged',
                  'PluggedOnly'],
    'Colorado' : ['PA',
                'pa'],
    'Florida' : ['P&A',
                 'DRY HOLE/P&A'],
    'Indiana' : ['Prsmd Plggd(I)',
                 'Plugd & Abandnd',
                 'Prsmd Plggd',
                 'Prsmd Plggd(I)',
                 'Inadqtly Plggd'],
    'Kansas' : ['OIL-P&A',
                'GAS-P&A',
                'EOR-P&A',
                'SWD-P&A',
                'OTHER-P&A(INJ or EOR)',
                'O&G-P&A',
                'OTHER-P&A(LH)',
                'CBM-P&A',
                'OTHER-P&A(STRAT)',
                'OTHER-P&A(CATH)',
                'OTHER-P&A()',
                'INJ-P&A',
                'OTHER-P&A(GAS-INJ)',
                'OTHER-P&A(OBS)',
                'OTHER-P&A(TA)',
                'OTHER-P&A(OIL&GAS-INJ)',
                'OTHER-P&A(GAS-STG)',
                'OTHER-P&A(GSW)',
                'OTHER-P&A(Plugged)',
                'OTHER(Plugged)',
                'OTHER-P&A(SWD-P&A)',
                'OTHER-P&A(Inj)',
                'OTHER-P&A(CLASS ONE (OLD))',
                'OTHER-P&A(2 OIL)'],
    # Kentucky : no category
    'Louisiana' : ['29', '30', '35', '90',
                  29, 30, 35, 90],
    'Michigan' : ['Plugging Approved',
                  'Plugging Completed'],
    'Missouri' : ['Plugged - Approved',
                  'Plugged - Not Approved'],
    'Montana' : ['P&A - Approved'],
    'Nebraska' : ['PA'],
    'Nevada' : ['P & A',
                'P&A',
                'P & A 7/27/95',
                'P & A (?)',
                'P & A 7/17/95'],
    'New Mexico' : ['Plugged (site released)',
                    'Plugged (not released)',
                    'Zone plugged (permanent)',
                    'Zone plugged (temporary)'],
    'New York' : ['PA',
                  'PB'],
    'North Dakota' : ['PA'],
    'Ohio' : ['PA'],
    'Oklahoma' : ['PA'],
    'Pennsylvania' : ['Plugged OG Well',
                      'DEP Plugged',
                      'Plugged Unverified',
                      'Plugged Mined Through'],
    'South Dakota' : ['Abandoned-Not Regulated',
                      'Plugged and Abandoned'],
    # Tennessee : no category
    'Texas' : [7,
               8,
               10,
               116,
               117,
               118,
               119,
               136,
               137,
               138,
               139,
               152,
               153,
               154,
               155,
               '7',
               '8',
               '10',
               '116',
               '117',
               '118',
               '119',
               '136',
               '137',
               '138',
               '139',
               '152',
               '153',
               '154',
               '155'],
    'Utah' : ['PA'],
    'West Virginia' : ['Plugged'],
    'Wyoming' : ['PA']
}

#%%

# =============================================================================
# 1. Import and standardize well status column in USGS dataset
# =============================================================================

# Ignore storage space warnings
warnings.filterwarnings("ignore")

# Import USGS dataset
usgs = pd.read_csv("USGS/US_orphaned_wells.csv")

# Clean USGS API number attribute
usgs['Well identifier'] = usgs['Well identifier'].str[4:-4]
usgs['Well identifier'] = usgs['Well identifier'].replace('-', '', regex=True).astype("string")
usgs['Well identifier'] = usgs['Well identifier'].replace(',', '', regex=True).astype("string")
usgs['Well identifier'] = usgs['Well identifier'].apply(lambda x: x.strip())

# Standardize well status attribute
usgs['Status'] = "ORPHANED"

#%%

# =============================================================================
# 2. Import and standardize well status column in FracTracker dataset
# =============================================================================

# Ignore storage space warnings
warnings.filterwarnings("ignore")

# Import FracTracker dataset
ft_prelim = pd.read_csv("FracTracker/full_dataset.csv")

# Add in Tennessee ds, which was accidentally not included in FT dataset
tn_ft = pd.read_csv("FracTracker/tennessee_wells_071624.csv")
ft = pd.concat([ft_prelim, tn_ft], ignore_index= True)

#%%
# Delete Kansas for now, replace it with the all wells dataset
ft = ft[ft['stusps'] != 'Kansas']

# Import Kansas straight from website
kansas_all_wells = pd.read_table('USGS/State_Downloads/ks_wells.txt', header=0, delimiter=",")
kansas_all_wells = kansas_all_wells[['API_NUMBER',
                                     'LEASE',
                                     'WELL',
                                     'TOWNSHIP',
                                     'RANGE',
                                     'SECTION',
                                     'STATUS2']]

# Clean FracTracker API number attribute
ft = ft.dropna(subset=['api_num'])
ft = ft[ft['api_num'] != 0000000000]
ft = ft[ft['api_num'] != '0000000000']
ft['api_num'] = ft.api_num.str.replace('-','')
ft['api_num'] = ft['api_num'].replace('-', '', regex=True).astype("string")
ft['api_num'] = ft['api_num'].replace(',', '', regex=True).astype("string")
ft['api_num'] = ft['api_num'].apply(lambda x: x.strip())
ft['api_num'] = ft['api_num'].astype(str)
ft = ft[ft['api_num'].str.len() >= 10]

# Take a peek at the status attributes for wells in FracTracker; will use later
ft_status = pd.DataFrame(ft.groupby('stusps').well_status.value_counts())

#%%

# Combine both dictionaries into one function for standardizing
def standardize_well_status(row):
    state = row['stusps']  
    status = row['well_status']
    
    # Check if the status belongs to the orphaned or plugged categories
    if state in state_status_dict and status in state_status_dict[state]:
        return 'ORPHANED'
    elif state in plugged_dict and status in plugged_dict[state]:
        return 'PLUGGED'
    return status  # If status doesn't match, keep the original

# Apply the mapping function to the 'well_status' column
ft['well_status'] = ft.apply(standardize_well_status, axis=1)

# Display the df to check the results
print(ft[['stusps', 'well_status']].head(n=10))

#%%

# =============================================================================
# 3. Work with FracTracker duplicate APIs
# =============================================================================

# Select only states of interest to make this more efficient
non_states = ['Arizona', 'Idaho', 'Illinois', 'Maryland', 'Oregon', 'Virginia',
              'Washington']
ft = ft[~ft['stusps'].isin(non_states)]
print("Starting length:", len(ft))

# Methodology:
# [STEP 1]: If they have the same api, well status, lat, and lon keep the last entry
# [STEP 2]: If they have the same api but different lat & lon, delete both
# [STEP 3]: If they have the same api, lat, and lon, but different well statuses:
#      [STEP 3a]: Keep the one listed as plugged
#      [STEP 3b]: If no well status is plugged, keep the one listed as orphaned
#      [STEP 3c]: If no well status is plugged or orphaned, keep the last entry

# [STEP 1]: Drop exact duplicates, keeping the last entry
ft = ft.drop_duplicates(subset=['api_num', 'well_status', 'latitude', 'longitude'], keep='last')
print("Step 1 Complete: Keep one version of exact duplicates")
print("Length after Step 1:", len(ft))
print('')

# [STEP 2]: Identify and delete APIs with multiple lat/lon entries
duplicate_api_mask = ft.duplicated(subset=['api_num'], keep=False)
api_groups = ft[duplicate_api_mask]
ft = ft[~duplicate_api_mask]
print("Step 2 Complete: Removed entries with the same API but different lat/lon.")
print("Length after Step 2:", len(ft))
print('')

#%%
# [STEP 3]: Handle same API, but diff status, prioritizing plugged or orphaned
# Define a function that prioritizes rows based on well status
def prioritize_status(group):
    # Check if "PLUGGED" status exists in the group
    if "PLUGGED" in group['well_status'].values:
        return group[group['well_status'] == "PLUGGED"].iloc[-1]
    # If no "PLUGGED" status, check for "ORPHANED"
    elif "ORPHANED" in group['well_status'].values:
        return group[group['well_status'] == "ORPHANED"].iloc[-1]
    # If neither "PLUGGED" nor "ORPHANED", keep the last entry
    else:
        return group.iloc[-1]

# Apply the prioritize_status function to each group of api nums
# Since we've deleted all api duplicates now, these are the ones that remain
# Therefore we only have to match on api
ft = ft.groupby(['api_num']).apply(prioritize_status)

# Reset the index after grouping to clean up the df structure
ft = ft.reset_index(drop=True)

# Summary after final filtering step
print("Step 3 Complete: Prioritized removal by well status")
print("Length after Step 23", len(ft))
print('')

#%%

# =============================================================================
# 4. Clean & process states that are completely included in FracTracker dataset
# =============================================================================

# USGS sourced these states from "all_wells" datasets, so I'm not going to download again

# Initialize Hauser df to contain list of 2024 orphaned wells
hauser_2024 = pd.DataFrame(columns=['api_10', 'lat', 'lon', 'state', 'county',
                                    'well_name', 'operator', 'well_status',
                                    'spud_date'])

# List of states that are completely included in ft
states = ['Alabama', 'Arkansas', 'Louisiana', 'Mississippi', 'Missouri', 
          'Nebraska', 'North Dakota', 'Ohio', 'Oklahoma', 'South Dakota']

# Add orphaned wells from these states to Hauser df
for state in states:
    # Work state-by-state
    print('------------------------------')
    print('Working on ' + state)
    
    # Selecting rows from each state
    ft_state = ft[ft['stusps'] == state]
    
    # Selecting only orphaned wells
    ft_orphaned = ft_state[ft_state['well_status'] == "ORPHANED"]
    print(state + ": " + str(len(ft_orphaned)) + " orphaned wells in 2024")
    print('')
    
    # Clean data to fit my desired fields
    data = []
    df = pd.DataFrame(data)
    df['api_10'] = ft_orphaned['api_num']
    df['lat'] = ft_orphaned['latitude']
    df['lon'] = ft_orphaned['longitude']
    df['state'] = ft_orphaned['stusps']
    df['county'] = ft_orphaned['county']
    df['well_name'] = ft_orphaned['well_name']
    df['operator'] = ft_orphaned['operator']
    df['well_status'] = ft_orphaned['well_status']
    df['spud_date'] = ft_orphaned['spud_date']
    
    # Append to hauser_2024
    hauser_2024 = pd.concat([hauser_2024, df], ignore_index=True)

#%%
# =============================================================================
# 5. Import states that required separate download
# =============================================================================

# USGS methodology is different from FracTracker's; need to download other datasets

# Ignore storage space warnings
warnings.filterwarnings("ignore") 

alaska = pd.read_csv("USGS/State_Downloads/Alaska_06282024.csv")
california = pd.read_csv("USGS/State_Downloads/California_07012024.csv")
colorado = pd.read_csv("USGS/State_Downloads/Colorado_06282024.csv")
co_wells = pd.read_csv("USGS/State_Downloads/Colorado_Wells_06282024.csv")
florida = pd.read_csv("USGS/State_Downloads/Florida_08142024.csv")
indiana = pd.read_csv("USGS/State_Downloads/Indiana_10092024.csv")
kansas = pd.read_csv("USGS/State_Downloads/Kansas_11102024.csv")
kentucky = pd.read_csv("USGS/State_Downloads/Kentucky_08262024.csv")
michigan = pd.read_csv("USGS/State_Downloads/Michigan_09232024.csv")
montana = pd.read_csv("USGS/State_Downloads/Montana_09132024.csv")
nevada = pd.read_csv("USGS/State_Downloads/Nevada_06212024.csv")
newmexico = pd.read_csv("USGS/State_Downloads/NewMexico_07152024.csv")
newyork = pd.read_csv("USGS/State_Downloads/NewYork_10092024.csv")
oklahoma = pd.read_csv("USGS/State_Downloads/Oklahoma_06212024.csv")
pennsylvania = pd.read_csv("USGS/State_Downloads/Pennsylvania_06212024.csv")
tennessee = pd.read_csv("USGS/State_Downloads/Tennessee_07012024.csv")
texas = pd.read_csv("USGS/State_Downloads/Texas_06212024.csv")
utah = pd.read_csv("USGS/State_Downloads/Utah_06212024.csv")
westvirginia = pd.read_csv("USGS/State_Downloads/WestVirginia_06212024.csv")
wyoming = pd.read_csv("USGS/State_Downloads/Wyoming_09132024.csv")

#%%

# =============================================================================
# 6. Specific state formatting
# =============================================================================

# Colorado
colorado = pd.merge(colorado, co_wells, left_on= 'Location ID', right_on = 'Loc_ID',  how='left')

# Indiana
# Define transformers for UTM Zones 16N and 17N
transformer_16N = Transformer.from_crs("EPSG:32616", "EPSG:4326")  # UTM Zone 16N to WGS84
transformer_17N = Transformer.from_crs("EPSG:32617", "EPSG:4326")  # UTM Zone 17N to WGS84

# Function to convert UTM to Latitude/Longitude with default Zone 16N
def utm_to_latlon_with_zone(easting, northing):
    # First transform using Zone 16
    lon_16, lat_16 = transformer_16N.transform(easting, northing)  # Correct order: (easting, northing)

    # Determine if the point might belong to Zone 17
    if -84 <= lon_16 < -78:  # Check if longitude suggests Zone 17
        lon_17, lat_17 = transformer_17N.transform(easting, northing)  # Correct order: (easting, northing)
        return lat_17, lon_17, 17
    return lat_16, lon_16, 16

# Apply the conversion and directly unpack the results into new columns
indiana[['Latitude', 'Longitude', 'UTM_Zone']] = indiana.apply(
    lambda row: pd.Series(utm_to_latlon_with_zone(row['Utmx'], row['Utmy'])), axis=1)

# Kansas
# Assign each well a bogus API number si they 
kansas['order'] = kansas.index + 1

# West Virginia
westvirginia_ft = ft[ft['stusps'] == 'West Virginia']
westvirginia['Well API'] = westvirginia['Well API'].astype("string")
westvirginia = pd.merge(westvirginia, westvirginia_ft, left_on= 'Well API', right_on = 'api_num',  how='left')

#%%

# =============================================================================
# 7. Define function to clean & process states that required separate download
# =============================================================================

# Define a function to standardize and combine datasets into one
def standardize_and_combine(states_data, state_fields_dict, required_fields):
    # Initialize an empty df for the final result
    combined_df = pd.DataFrame()  
    
    for state_name, state_df in states_data.items():
        print('---------------------')
        print('Cleaning ' + state_name)
        # Create an empty df for the current state's cleaned data
        clean_df = pd.DataFrame()
        
        # Loop through the required fields and map them to the state-specific columns
        for std_col in required_fields:
            # Use .get() to avoid KeyErrors when a field is missing
            state_col = state_fields_dict[state_name].get(std_col, None)
            if state_col in state_df.columns:
                clean_df[std_col] = state_df[state_col]
            else:
                # Fill missing columns with NaN
                clean_df[std_col] = np.nan  
        
        # Add the state name as a new column
        clean_df['state'] = state_name
        
        print(state_name + ": " + str(len(clean_df)) + " orphaned wells in 2024")
        print('')
        
        # Append the cleaned data to the combined DataFrame
        combined_df = pd.concat([combined_df, clean_df], ignore_index=True)
    
    return combined_df

#%%

# =============================================================================
# 8. Define parameters for function
# =============================================================================

# Dictionary of state dfs that I need
states_data = {
    'Alaska' : pd.DataFrame(alaska),
    'California' : pd.DataFrame(california),
    'Colorado' : pd.DataFrame(colorado),
    'Florida' : pd.DataFrame(florida),
    'Indiana' : pd.DataFrame(indiana),
    'Kansas' : pd.DataFrame(kansas),
    'Kentucky' : pd.DataFrame(kentucky),
    'Michigan' : pd.DataFrame(michigan),
    'Montana' : pd.DataFrame(montana),
    'Nevada' : pd.DataFrame(nevada),
    'New Mexico' : pd.DataFrame(newmexico),
    'New York' : pd.DataFrame(newyork),
    'Oklahoma' : pd.DataFrame(oklahoma),
    'Pennsylvania' : pd.DataFrame(pennsylvania),
    'Tennessee' : pd.DataFrame(tennessee),
    'Texas' : pd.DataFrame(texas),
    'Utah' : pd.DataFrame(utah),
    'West Virginia' : pd.DataFrame(westvirginia),
    'Wyoming' : pd.DataFrame(wyoming)
    }

# Define the column mapping for each state
state_fields_dict = { 
                    'Alaska' : 
                     {'api_10' : 'US Well ID/API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      'state' : 'State',
                      #'county'
                      'well_name' : 'Well Name',
                      'operator' : 'Surface Managing Entity Name'
                      #'well_status' 
                      #'spud_date' 
                      }, 
                     
                     'California' :
                     {'api_10' : 'API 10',
                      'lat' : 'lat',
                      'lon' : 'lon',
                      #'state'
                      'county' : 'County',
                      'well_name' : 'Well Designation',
                      'operator' : 'Operator Name'
                      #'well_status'
                      #'spud_date'
                      },
                     
                     'Colorado' : 
                     {'api_10' : 'API_Label',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                       #'state'
                       'county' : 'County',
                       'well_name' : 'Well_Title',
                       'operator' : 'Operator',
                       'well_status' : 'Facil_Stat',
                       'spud_date' : 'Spud_Date'
                       },
                     
                     'Florida' :
                     {'api_10' : 'API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                       #'state'
                       'county' : 'COUNTY',
                       'well_name' : 'WELL_NAME',
                       #'operator'
                       'well_status' : 'Current Status',
                      },
                     
                     'Indiana' :
                     {'api_10' : 'Permit_Number', # Indiana doesn't use API
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                       #'state'
                       'county' : 'County',
                       'well_name' : 'Lease_Name',
                       'operator' : 'Operator_Name',
                       'well_status' : 'Status',
                       'spud_date' : 'Well_Number' # I'm cheating and putting well number here
                      },
                     
                     'Kansas' :
                     {'api_10' : 'order', # Kansas doesn't use API; made bogus ones so they don't get dropped during cleaning
                      'lat' : 'Twp.', # Cheating and putting twp here
                      'lon' : 'Sect.', # Cheating and putting sect here
                      #'state'
                      'county' : 'County',
                      'well_name' : 'Lease Name',
                      'operator' : 'Well Number', # I'm cheating and putting well number here
                      #'well_status'
                      'spud_date' : 'Rng.' # Cheating and putting rng here
                      },
                     
                     'Kentucky' :
                     {'api_10' : ' API No ',
                      'lat' : 'LAT',
                      'lon' : 'LONG',
                      #'state'
                      'county' : 'County',
                      'well_name' : 'Well Name',
                      #'operator' 
                      'well_status' : 'Well Type'
                      #'spud_date'
                      },
                     
                     'Michigan' :
                     {'api_10' : 'API #',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state'
                      'county' : 'Cnty',
                      'well_name' : 'Name',
                      'operator' : 'Operator',
                      'well_status' : 'Status'
                      #'spud_date'
                      },
                     
                     'Montana' :
                     {'api_10' : 'API #',
                      'lat' : 'LAT',
                      'lon' : 'LON',
                      #'state'
                      'county' : 'County',
                      'well_name' : 'Well_Nm',
                      'operator' : 'CoName'
                      #'well_status' 
                      #'spud_date'
                      },
                     
                     'Nevada' :
                     {'api_10' : 'apino',
                      'lat' : 'latdegree',
                      'lon' : 'longdegree',
                      'state' : 'state_',
                      'county' : 'county',
                      'well_name' : 'wellname',
                      'operator' : 'operator_',
                      'well_status' : 'status',
                      'spud_date' : 'spuddatetime'
                      },
                     
                     'New Mexico' :
                     {'api_10' : 'API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state' 
                      #'county'
                      'well_name' : 'Well Name',
                      'operator' : 'Current Operator',
                      'well_status' : 'Status',
                      'spud_date' : 'Spud Date'
                      },
                     
                     'New York' :
                     {'api_10' : 'API WELL NUMBER',
                      'lat' : 'SURFACE LATITUDE',
                      'lon' : 'SURFACE LONGITUDE',
                      #'state' 
                      'county' : 'COUNTY',
                      'well_name' : 'WELL NAME',
                      'operator' : 'COMPANY NAME',
                      'well_status' : 'WELL STATUS',
                      # 'spud_date'
                      },
                     
                     'Oklahoma' :
                     {'api_10' : 'API',
                      'lat' : 'Y',
                      'lon' : 'X',
                      #'state' 
                      'county' : 'CountyName',
                      'well_name' : 'WellName',
                      'operator' : 'OperatorName',
                      'well_status' : 'WellStatus'
                      #'spud_date'
                      },
                     
                     'Pennsylvania' :
                     {'api_10' : 'API',
                      'lat' : 'LATITUDE_DECIMAL',
                      'lon' : 'LONGITUDE_DECIMAL',
                      #'state' 
                      'county' : 'COUNTY',
                      'well_name' : 'FARM_NAME',
                      'operator' : 'OPERATOR',
                      'well_status' : 'WELL_STATUS'
                      #'spud_date'
                      },
                     
                     'Tennessee' :
                     {'api_10' : 'API',
                      'lat' : 'LAT',
                      'lon' : 'LONG',
                      #'state' 
                      'county' : 'COUNTYNAME',
                      'well_name' : 'WELLNAME',
                      'operator' : 'OPNAME',
                      'well_status' : 'WELL_STATUS'
                      #'spud_date'
                      },
                     
                     'Texas' :
                     {'api_10' : 'API',
                      'lat' : 'latitude',
                      'lon' : 'longitude',
                      #'state' 
                      'county' : 'COUNTY_NAME',
                      #'well_name'
                      'operator' : 'OPERATOR_NAME'
                      #'well_status'
                      #'spud_date'
                      },
                     
                     'Utah' :
                     {'api_10' : 'API',
                      'lat' : 'Latitude',
                      'lon' : 'Longitude',
                      #'state' 
                      'county' : 'CountyName',
                      'well_name' : 'WellName',
                      'operator' : 'Operator',
                      'well_status' : 'wellstatus'
                      #'spud_date'
                      },
                     
                     'West Virginia' :
                     {'api_10' : 'Well API',
                      'lat' : 'latitude',
                      'lon' : 'longitude',
                      'state' : 'stusps', 
                      'county' : 'county',
                      'well_name' : 'Well Number',
                      'operator' : 'Surface Owner',
                      'well_status' : 'Well Status'
                      #'spud_date'
                      },
                     
                     'Wyoming' :
                     {'api_10' : 'Apino',
                      'lat' : 'Lat',
                      'lon' : 'Lon',
                      #'state' 
                      #'county'
                      'well_name' : 'Wellname',
                      'operator' : 'Company',
                      'well_status' : 'F2Status'
                      #'spud_date'
                      },
                     }

# Define the required fields for Hauser df
required_fields = ['api_10', 'lat', 'lon', 'state', 'county', 'well_name', 'operator', 'well_status', 'spud_date']

#%%
# =============================================================================
# 9. Call standardize and combine function
# =============================================================================

# Call the function
hauser_add_on = standardize_and_combine(states_data, state_fields_dict, required_fields)
print('-----------------------------------------')

# Combine datasets
hauser_2024_combo = pd.concat([hauser_2024, hauser_add_on], ignore_index=True, sort=False)

# Delete duplicate wells from each state
hauser_2024 = hauser_2024_combo.drop_duplicates(subset='api_10', keep=False)

#%%
# =============================================================================
# 10. Clean Hauser df
# =============================================================================

# Delete rows with no API number or duplicate API numbers 
hauser_2024 = hauser_2024.dropna(subset=['api_10']) 

# Delete those with missing lat/lons
hauser_2024 = hauser_2024.dropna(subset=['lat'])
hauser_2024 = hauser_2024.dropna(subset=['lon'])
hauser_2024 = hauser_2024[hauser_2024['lat'] != 0]
hauser_2024 = hauser_2024[hauser_2024['lon'] != 0]
hauser_2024 = hauser_2024[hauser_2024['lat'] != 'nan']
hauser_2024 = hauser_2024[hauser_2024['lon'] != 'nan']

# Convert lat & lon to numeric dtypes
hauser_2024['lat'] = pd.to_numeric(hauser_2024['lat'], errors='coerce')
hauser_2024['lon'] = pd.to_numeric(hauser_2024['lon'], errors='coerce')

# Make sure all lons are negative
# (lats are within bounds)
hauser_2024['lon'] = hauser_2024['lon'].abs()
hauser_2024['lon'] = hauser_2024['lon']*-1 

# Make API consistent
hauser_2024['api_10'] = hauser_2024['api_10'].replace('-', '', regex=True).astype("string")
hauser_2024['api_10'] = hauser_2024['api_10'].replace(',', '', regex=True).astype("string")
hauser_2024['api_10'] = hauser_2024['api_10'].apply(lambda x: x.strip())
hauser_2024['api_10'] = hauser_2024['api_10'].apply(lambda x: x[:10] if pd.notna(x) and len(x) > 10 else x)

# Apply special formatting for relevant states
hauser_2024.loc[hauser_2024['state'] == 'California', 'api_10'] = hauser_2024.loc[hauser_2024['state'] == 'California', 'api_10'].str.zfill(10)
hauser_2024.loc[hauser_2024['state'] == 'Florida', 'api_10'] = hauser_2024.loc[hauser_2024['state'] == 'Florida', 'api_10'].str.zfill(10)
hauser_2024.loc[hauser_2024['state'] == 'Pennsylvania', 'api_10'] = '37' + hauser_2024.loc[hauser_2024['state'] == 'Pennsylvania', 'api_10']    
hauser_2024.loc[hauser_2024['state'] == 'Tennessee', 'api_10'] = '41' + hauser_2024.loc[hauser_2024['state'] == 'Tennessee', 'api_10']
hauser_2024.loc[hauser_2024['state'] == 'Texas', 'api_10'] = '42' + hauser_2024.loc[hauser_2024['state'] == 'Texas', 'api_10']  
hauser_2024.loc[hauser_2024['state'] == 'Wyoming', 'api_10'] = '490' + hauser_2024.loc[hauser_2024['state'] == 'Wyoming', 'api_10']  

# Make spud date consistent (datetime)
#hauser_2024['spud_date'] = pd.to_datetime(hauser_2024['spud_date'], errors='coerce')

# Add state abbreviation column
#List of states
state2abbrev = {'Alaska': 'AK',
                'Alabama': 'AL',
                'Arkansas': 'AR',
                'California': 'CA',
                'Florida': 'FL',
                'Indiana': 'IN',
                'Kansas': 'KS',
                'Kentucky': 'KY',
                'Louisiana': 'LA',
                'Michigan': 'MI',
                'Missouri': 'MO',
                'Mississippi': 'MS',
                'Montana': 'MT',
                'North Dakota': 'ND',
                'Nebraska': 'NE',
                'New Mexico': 'NM',
                'Nevada': 'NV',
                'New York': 'NY',
                'Ohio': 'OH',
                'Oklahoma': 'OK',
                'Pennsylvania': 'PA',
                'South Dakota': 'SD',
                'Tennessee': 'TN',
                'Texas': 'TX',
                'Utah': 'UT',
                'Virginia': 'VA',
                'West Virginia': 'WV',
                'Wyoming': 'WY'}
hauser_2024['st_abbrev'] = hauser_2024['state'].map(state2abbrev)


#%%
# =============================================================================
# 10. Delete those who are listed as plugged in FracTracker 
# =============================================================================

# Using API: if a well is listed as plugged in FracTracker, remove it from Hauser_2024
# Filter FT dataset to only include plugged wells
plugged_wells_ft = ft[ft['well_status'] == 'PLUGGED']
plugged_wells_ft = plugged_wells_ft[['stusps', 'api_num', 'operator', 'well_name']]
plugged_wells_ks = kansas_all_wells[kansas_all_wells['STATUS2'] == 'Plugged and Abandoned']

# Make sure both are the same datatype
plugged_wells_ft['api_num'] = plugged_wells_ft['api_num'].astype("string")
hauser_2024['api_10'] = hauser_2024['api_10'].astype("string")

# Split the data into Indiana and other datasets for different merge conditions
indiana_wells = hauser_2024[hauser_2024['state'] == 'Indiana']
kansas_wells = hauser_2024[hauser_2024['state'] == 'Kansas']
other_wells = hauser_2024[hauser_2024['state'] != 'Indiana']
other_wells = other_wells[other_wells['state'] != 'Kansas']

# Merge Indiana wells on operator name and lease name
indiana_merged = pd.merge(indiana_wells, plugged_wells_ft,
                          left_on=['operator', 'well_name'], 
                          right_on=['operator', 'well_name'],
                          how='left', indicator=True)

# Merge Kansas wells on ____
kansas_merged = pd.merge(kansas_wells, plugged_wells_ks,
                          left_on=['well_name', 'operator', 'lat', 'lon', 'spud_date'], 
                          right_on=['LEASE', 'WELL', 'TOWNSHIP', 'RANGE', 'SECTION'],
                          how='left', indicator=True)

# Merge other wells on API numbers
other_merged = pd.merge(other_wells, plugged_wells_ft,
                        left_on='api_10', right_on='api_num', how='left',
                        indicator=True)

# Drop the temporary merge columns & rename duplicates
indiana_merged = indiana_merged.drop(['api_num', 'stusps', 'latitude', 'longitude'], axis=1, errors='ignore')
kansas_merged = kansas_merged.drop(['API_NUMBER', 'LEASE', 'WELL', 'TOWNSHIP', 'RANGE', 'SECTION'], axis=1, errors='ignore')
other_merged = other_merged.drop(['stusps', 'api_num', 'operator_y', 'well_name_y',], axis=1, errors='ignore')
other_merged = other_merged.rename(columns={'well_name_x': 'well_name', 'operator_x': 'operator'})

# Combine both merged datasets
hauser_2024f = pd.concat([indiana_merged, kansas_merged, other_merged])

# Create DataFrame of wells actually plugged in FracTracker
actually_plugged = hauser_2024f[hauser_2024f['_merge'] == 'both']
print(actually_plugged.groupby('state').size().reset_index(name='Actually_plugged'))

# Remove plugged wells from Hauser_2024 based on merge results
hauser_2024f = hauser_2024f[hauser_2024f['_merge'] == 'left_only']


# Drop the temporary merge columns
hauser_2024f = hauser_2024f.drop(['_merge'], axis=1, errors='ignore')

# Display the final grouped count by state
print('-----------------------------------------')
print(hauser_2024f.groupby('state').size().reset_index(name='Hauser_well_count'))
print(hauser_2024f.columns)
hauser_2024_grouped = hauser_2024f.groupby('state').size().reset_index(name='Hauser_well_count')


#%%

# =============================================================================
# =============================================================================
# =============================================================================
# AIM 2 : HOW MANY WELLS HAVE BECOME ORPHANED SINCE THE USGS REPORT?
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Compare APIs in Hauser_2024 to USGS
# =============================================================================

# Separate Indiana, Kansas and other wells
indiana_wells = hauser_2024f[hauser_2024f['state'] == 'Indiana']
kansas_wells = hauser_2024f[hauser_2024f['state'] == 'Kansas']
other_wells = hauser_2024f[hauser_2024f['state'] != 'Indiana']
other_wells = other_wells[other_wells['state'] != 'Kansas']

# Ensure columns to compare have the same data type
usgs[['County', 'Well name', 'Well number']] = usgs[['County', 'Well name', 'Well number']].astype("string")
indiana_wells[['well_name', 'spud_date']] = indiana_wells[['well_name', 'spud_date']].astype("string")
kansas_wells[['county', 'well_name', 'operator']] = kansas_wells[['county', 'well_name', 'operator']].astype("string")

# Find Indiana wells that are not in USGS based on well name and number
indiana_newly_orphaned = indiana_wells[
    ~indiana_wells[['well_name', 'spud_date']].apply(tuple, axis=1).isin(
        usgs[['Well name', 'Well number']].apply(tuple, axis=1))]

# Find Kansas wells that are not in USGS based on county, well name, and well number
kansas_newly_orphaned = kansas_wells[
    ~kansas_wells[['county', 'well_name', 'operator']].apply(tuple, axis=1).isin(
        usgs[['County', 'Well name', 'Well number']].apply(tuple, axis=1))]

# Find non-Indiana non-Kansas wells that are not in USGS based on API
other_newly_orphaned = other_wells[~other_wells['api_10'].isin(usgs['Well identifier'])]

# Concatenate the results
newly_orphaned = pd.concat([indiana_newly_orphaned, 
                            kansas_newly_orphaned, 
                            other_newly_orphaned])

# Get a count of newly orphaned wells by state
print('-----------------------------------------')
print(newly_orphaned.groupby('state').size().reset_index(name='new_orphaned_well_count'))
newly_orphaned_grouped = newly_orphaned.groupby('state').size().reset_index(name='new_orphaned_well_count')

#%%

# =============================================================================
# 2. Make hauser_status column
# =============================================================================
# Add default "Orphaned since USGS" status to all wells in hauser_2024f
hauser_2024f['hauser_status'] = 'Orphaned since USGS'

# Update status to "Newly orphaned" for Indiana wells in newly_orphaned
hauser_2024f.loc[
    hauser_2024f[['state', 'well_name', 'spud_date']].apply(tuple, axis=1).isin(
        indiana_newly_orphaned[['state', 'well_name', 'spud_date']].apply(tuple, axis=1)
    ),
    'hauser_status'
] = 'Newly orphaned'

# Update status to "Newly orphaned" for Kansas wells in newly_orphaned
hauser_2024f.loc[
    hauser_2024f[['state', 'county', 'well_name', 'operator']].apply(tuple, axis=1).isin(
        kansas_newly_orphaned[['state', 'county', 'well_name', 'operator']].apply(tuple, axis=1)),
    'hauser_status'] = 'Newly orphaned'

# Update status to "Newly orphaned" for other states based on API match
hauser_2024f.loc[
    hauser_2024f['api_10'].isin(other_newly_orphaned['api_10']),
    'hauser_status'] = 'Newly orphaned'

# Check the final DataFrame
print(hauser_2024f[['state', 'hauser_status']].value_counts())


#%%

# =============================================================================
# =============================================================================
# =============================================================================
# AIM 3 : HOW MANY WELLS HAVE BEEN PLUGGED SINCE THE USGS REPORT?
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Compare APIs in USGS to FT Plugged
# =============================================================================
 
# THIS WON'T WORK FOR INDIANA OR KANSAS

# Find APIs in USGS but not in Hauser 2024
newly_plugged = usgs[~usgs['Well identifier'].isin(hauser_2024['api_10'])]

# From this, drop APIs that have a status other than "PLUGGED" in ft
newly_plugged = newly_plugged[newly_plugged['Well identifier'].isin(plugged_wells_ft['api_num'])]

# From this, drop APIs that aren't in hauser_2024 bc they're actually plugged while currently listed as orphaned
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].isin(actually_plugged['api_10'])]

# From this, drop APIs that are fake (USGS assigned value)
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].astype(str).str.startswith('ID')]
newly_plugged = newly_plugged[~newly_plugged['Well identifier'].astype(str).str.startswith('D')]

# Separate process for Kansas
ks_usgs = usgs[usgs['State'] == 'Kansas']
# Find wells in USGS but not in Hauser 2024 based on County, well_name, and operator
ks_newly_plugged = ks_usgs[
    ~ks_usgs.set_index(['County', 'Well name', 'Well number']).index.isin(
        hauser_2024.set_index(['county', 'well_name', 'operator']).index)]

# From this, drop wells that have a status other than "PLUGGED" in all wells ds
ks_newly_plugged = ks_newly_plugged[
    ks_newly_plugged.set_index(['Well name', 'Well number', 'Township', 'Range', 'Section']).index.isin(
        plugged_wells_ks.set_index(['LEASE', 'WELL', 'TOWNSHIP', 'RANGE', 'SECTION']).index)]


# Concat Kansas data
newly_plugged = pd.concat([newly_plugged, ks_newly_plugged], ignore_index= True)

# View
newly_plugged_grouped = newly_plugged.groupby('State').size().reset_index(name='since_plugged_well_count')
print('-----------------------------------------')
print(newly_plugged_grouped) 

#%%

# =============================================================================
# =============================================================================
# =============================================================================
# VISUALIZE & EXPORT
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Validate that wells are within their specified states
# ============================================================================= 

# REMOVE KANSAS
hauser_2024f = hauser_2024f[hauser_2024f.state != 'Kansas']

# Fix Indiana attributes
hauser_2024f.loc[hauser_2024f['state'] == 'Indiana', 'spud_date'] = pd.NA

# Convert to gdf
hauser_2024_gdf = gpd.GeoDataFrame(hauser_2024f,
                       geometry=gpd.points_from_xy(hauser_2024f.lon, hauser_2024f.lat),
                       crs="EPSG:4326")

from pygris import states
import us  # us library provides mappings for state names and abbreviations

# Retrieve all state boundaries
state_boundaries = states()

# Create a dictionary to map state names to abbreviations
state_name_to_abbr = {state.name.upper(): state.abbr for state in us.states.STATES}

def validate_points_in_state(gdf):
    # Standardize and map full state names to abbreviations
    gdf['state_abbr'] = gdf['state'].str.upper().map(state_name_to_abbr)
    
    # Initialize an empty list to store validation results
    validation_results = []
    
    for idx, row in gdf.iterrows():
        # Filter state boundaries for the claimed state
        state_geom = state_boundaries[state_boundaries['STUSPS'] == row['state_abbr']].geometry
        
        # Check if point is within the claimed state boundary
        if not state_geom.empty and row['geometry'].within(state_geom.iloc[0]):
            validation_results.append(True)
        else:
            validation_results.append(False)
    
    # Add validation results to the original geodataframe
    gdf['is_within_claimed_state'] = validation_results
    # Drop the temporary abbreviation column
    gdf.drop(columns=['state_abbr'], inplace=True)
    return gdf

hauser_2024_gdf = validate_points_in_state(hauser_2024_gdf) 

#%%

# =============================================================================
# 2. Map all pts
# ============================================================================= 

from pygris import states
from pygris.utils import shift_geometry

us = states(cb = True, resolution = "20m")
us_rescaled = shift_geometry(us)

orphans_rescaled = shift_geometry(hauser_2024_gdf)
fig, ax = plt.subplots()

us_rescaled.plot(ax = ax, color = "grey") 
orphans_rescaled.plot(ax = ax, color = "black", marker='o', markersize=2)

# Set axis limits for the contiguous US
ax.set_xlim(us_rescaled.total_bounds[0], us_rescaled.total_bounds[2])
ax.set_ylim(us_rescaled.total_bounds[1], us_rescaled.total_bounds[3])

# Add a title for context
ax.set_title("Map of Orphaned Wells Across the United States")

# Show the plot
plt.show()

#%%

# =============================================================================
# 3. Export 
# ============================================================================= 

# Change directory
os.chdir('/Users/gracehauser/Desktop/Thesis/00 - Data') 

# NOTE: NOT CONTAINING SOME STATES RIGHT NOW

# Hauser final ds
hauser_2024_gdf.to_file('Hauser_2024/hauser_2024.shp', driver='ESRI Shapefile')

# Newly plugged ds
newly_plugged_gdf = gpd.GeoDataFrame(newly_plugged,
                        geometry=gpd.points_from_xy(newly_plugged.Longitude, newly_plugged.Latitude),
                        crs="EPSG:4326")
newly_plugged_gdf.to_file('Newly_Plugged/newly_plugged.shp', driver='ESRI Shapefile')

# Newly orphaned ds
newly_orphaned_gdf = gpd.GeoDataFrame(newly_orphaned,
                        geometry=gpd.points_from_xy(newly_orphaned.lon, newly_orphaned.lat),
                        crs="EPSG:4326")
newly_orphaned_gdf.to_file('Newly_Orphaned/newly_orphaned.shp', driver='ESRI Shapefile') 



