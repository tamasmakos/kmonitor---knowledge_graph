import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_dataset(input_file: str) -> pd.DataFrame:
    # Read data
    data = pd.read_csv(input_file)
    
    # Filter for awarded contracts only
    data = data[data['Szerződés/rész odaítélésre került'] == 'Igen']
    logger.info(f"Filtered for awarded contracts: {len(data)} rows")
    #drop 
    
    # Drop EUR currency rows
    data = data[data['A beszerzés végleges összértéke pénznem'] != 'EUR']
    data = data.dropna(subset=['A beszerzés végleges összértéke'])
    logger.info(f"Filtered out EUR contracts: {len(data)} rows remaining")
    
    # Process dates
    data['Hirdetmény közzétételének dátuma'] = pd.to_datetime(data['Hirdetmény közzétételének dátuma'])
    data['Szerződés megkötésének dátuma'] = pd.to_datetime(data['Szerződés megkötésének dátuma'])
    data['Hirdetmény és szerződés megkötés között eltelt idő'] = (
        data['Szerződés megkötésének dátuma'] - data['Hirdetmény közzétételének dátuma']
    )
    
    # Add unique ID
    data['unique_id'] = data.index
    
    logger.info(f"Cleaning completed. Final shape: {data.shape}")

    #write to csv
    data.to_csv('cleaned_data.csv', index=False)

    return data



