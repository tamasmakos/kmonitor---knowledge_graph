# Import required libraries
import logging
import pandas as pd
import recordlinkage
from recordlinkage.preprocessing import clean
from typing import Dict, Any, Tuple
import numpy as np
from collections import defaultdict
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)

def preprocess_tax_id(tax_id: str) -> str:
    """Clean and standardize tax ID format"""
    if pd.isna(tax_id) or tax_id == '':
        return ''
    
    # Convert to string if not already
    tax_id = str(tax_id)
    
    # Remove any non-numeric characters
    tax_id = ''.join(filter(str.isdigit, tax_id))
    
    return tax_id

def preprocess_company_name(name: str) -> str:
    """Clean and standardize a single company name by removing Hungarian company type suffixes"""
    if pd.isna(name) or name == '':
        return ''
    
    name = str(name).strip().lower()
    
    # Remove common Hungarian legal entity suffixes and their variations
    suffixes = [
        'kft.', 'kft', 'korlátolt felelősségű társaság',
        'zrt.', 'zrt', 'zártkörűen működő részvénytársaság',
        'nyrt.', 'nyrt', 'nyilvánosan működő részvénytársaság',
        'bt.', 'bt', 'betéti társaság',
        'kkt.', 'kkt', 'közkereseti társaság',
        'nonprofit', 'nkft.', 'nkft',
        'kht.', 'kht', 'közhasznú társaság',
        'rt.', 'rt', 'részvénytársaság',
        'ev.', 'ev', 'egyéni vállalkozó',
        'gmk.', 'gmk', 'gazdasági munkaközösség',
        'szöv.', 'szöv', 'szövetkezet',
        'kk.', 'kk', 'közös vállalat',
        'kv.', 'kv', 'költségvetési szerv',
        'e.v.', 'egyéni cég', 'ec.',
        'egyesület', 'alapítvány', 'önkormányzat'
    ]
    
    # Sort suffixes by length (longest first) to avoid partial matches
    suffixes = sorted(suffixes, key=len, reverse=True)
    
    # Remove suffixes and extra whitespace
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            break
    
    # Convert single string to Series for recordlinkage.clean()
    name_series = pd.Series([name])
    cleaned_series = clean(name_series)
    
    # Get the cleaned string back from the series
    cleaned_name = cleaned_series.iloc[0] if not cleaned_series.empty else ''
    
    return cleaned_name.strip()

def expand_multiple_entities(df: pd.DataFrame, name_col: str, tax_id_col: str) -> pd.DataFrame:
    """Expand rows with multiple entities into separate rows"""
    expanded_rows = []
    
    for _, row in df.iterrows():
        # Handle NaN values
        if pd.isna(row[name_col]):
            expanded_rows.append(row)
            continue
            
        # Convert to string and split
        names = str(row[name_col]).split('|') if '|' in str(row[name_col]) else [str(row[name_col])]
        
        # Create a row for each entity
        for name in names:
            new_row = row.copy()
            new_row[name_col] = name.strip()
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

def resolve_entities(data: pd.DataFrame, entity_columns: Dict, similarity_threshold: float = 0.90) -> Tuple[pd.DataFrame, Dict]:
    """
    Resolve entities using both tax ID and name similarity with stricter rules:
    1. If tax IDs match, also check name similarity
    2. Only merge if name similarity is above threshold when tax IDs match
    """
    logger.info("Starting entity resolution...")
    
    resolved_data = data.copy()
    resolution_stats = {
        'matches': [],
        'total_entities': 0,
        'merged_entities': 0
    }

    # Process each entity type (issuer and winner)
    for entity_type, columns in entity_columns.items():
        name_col = columns['name']
        tax_id_col = columns['tax_id']
        
        # Create pairs of potential matches based on tax ID
        tax_id_matches = defaultdict(list)
        for idx, row in resolved_data.iterrows():
            if pd.notna(row[tax_id_col]):
                tax_id_matches[str(row[tax_id_col])].append(row[name_col])
        
        # Process tax ID matches with name similarity check
        name_mappings = {}
        for tax_id, names in tax_id_matches.items():
            unique_names = list(set(names))
            if len(unique_names) > 1:
                for i in range(len(unique_names)):
                    for j in range(i + 1, len(unique_names)):
                        name1 = unique_names[i]
                        name2 = unique_names[j]
                        
                        # Calculate name similarity
                        similarity = calculate_name_similarity(name1, name2)
                        
                        # Only merge if similarity is above threshold
                        if similarity >= similarity_threshold:
                            canonical_name = min(name1, name2, key=len)
                            name_mappings[name1] = canonical_name
                            name_mappings[name2] = canonical_name
                            
                            # Log the match
                            resolution_stats['matches'].append({
                                'entity_type': entity_type,
                                'name1': name1,
                                'name2': name2,
                                'tax_id': tax_id,
                                'similarity': similarity,
                                'match_type': 'tax_id_and_name'
                            })
                            logger.info(f"Merged {name1} and {name2} (tax_id: {tax_id}, similarity: {similarity:.2f})")
        
        # Apply mappings
        resolved_data[name_col] = resolved_data[name_col].map(lambda x: name_mappings.get(x, x))
        
        # Update stats
        resolution_stats['total_entities'] = len(resolved_data[name_col].unique())
        resolution_stats['merged_entities'] = len(name_mappings)
    
    logger.info(f"Entity resolution complete. Merged {resolution_stats['merged_entities']} entities.")
    return resolved_data, resolution_stats

def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using multiple metrics"""
    # Clean and normalize names
    name1 = clean_name(name1)
    name2 = clean_name(name2)
    
    # Calculate different similarity metrics
    levenshtein_ratio = fuzz.ratio(name1, name2) / 100
    token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100
    
    # Return the average of the metrics
    return (levenshtein_ratio + token_sort_ratio) / 2

def clean_name(name: str) -> str:
    """Clean and normalize entity names"""
    if not isinstance(name, str):
        return str(name)
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove common abbreviations and legal forms
    replacements = {
        'zrt.': 'zrt',
        'kft.': 'kft',
        'bt.': 'bt',
        'nyrt.': 'nyrt',
        'rt.': 'rt'
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name

