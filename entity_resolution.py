# Import required libraries
import logging
import pandas as pd
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def resolve_entities(df: pd.DataFrame, entity_cols: Dict[str, Dict[str, str]], similarity_threshold: float = 0.85) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resolve entities in the dataframe using only tax ID matching
    """
    logger.info(f"Starting entity resolution with data shape: {df.shape}")
    
    resolution_stats = {
        'total_entities': 0,
        'total_merged': 0,
        'by_type': {}
    }
    
    resolved_df = df.copy()
    
    for entity_type, columns in entity_cols.items():
        logger.info(f"\nProcessing {entity_type} entities...")
        
        name_col = columns['name']
        tax_id_col = columns['tax_id']
        
        # Log unique counts before resolution
        logger.info(f"Before resolution - unique {entity_type} names: {resolved_df[name_col].nunique()}")
        
        # Extract unique entities
        entities = df[[name_col, tax_id_col]].drop_duplicates()
        original_count = len(entities)
        logger.info(f"Found {original_count} unique {entity_type} entities")
        
        # Find tax ID matches
        tax_id_matches = []
        valid_entities = entities[entities[tax_id_col].notna() & (entities[tax_id_col] != '')]
        
        if not valid_entities.empty:
            tax_id_groups = valid_entities.groupby(tax_id_col)
            
            for tax_id, group in tax_id_groups:
                if len(group) > 1:
                    primary = group.iloc[0][name_col]
                    for _, row in group.iloc[1:].iterrows():
                        tax_id_matches.append({
                            'primary_name': primary,
                            'primary_tax_id': tax_id,
                            'merged_name': row[name_col],
                            'merged_tax_id': tax_id
                        })
            
            if tax_id_matches:
                # Create mapping dictionary
                mapping = {}
                for match in tax_id_matches:
                    mapping[match['merged_name']] = match['primary_name']
                
                # Apply mapping to dataframe
                resolved_df[name_col] = resolved_df[name_col].map(mapping).fillna(resolved_df[name_col])
                
                # Log examples and statistics
                logger.info(f"Found {len(tax_id_matches)} tax ID matches")
                logger.info("\nExample matches (showing up to 3):")
                for match in tax_id_matches[:3]:
                    logger.info(f"  {match['merged_name']} -> {match['primary_name']} (same tax ID: {match['primary_tax_id']})")
                
                # Update statistics
                type_stats = {
                    'original_count': original_count,
                    'merged_count': len(tax_id_matches),
                    'final_count': len(resolved_df[name_col].unique())
                }
                resolution_stats['by_type'][entity_type] = type_stats
                resolution_stats['total_merged'] += len(tax_id_matches)
            else:
                logger.info("No tax ID matches found")
        
        resolution_stats['total_entities'] += original_count
        
        # Log unique counts after resolution
        logger.info(f"After resolution - unique {entity_type} names: {resolved_df[name_col].nunique()}")
    
    logger.info(f"Final resolved data shape: {resolved_df.shape}")
    logger.info("\n=== Entity Resolution Summary ===")
    logger.info(f"Total unique entities: {resolution_stats['total_entities']}")
    logger.info(f"Total entities merged: {resolution_stats['total_merged']}")
    
    for entity_type, stats in resolution_stats['by_type'].items():
        logger.info(f"\n{entity_type}:")
        logger.info(f"  Original count: {stats['original_count']}")
        logger.info(f"  Merged count: {stats['merged_count']}")
        logger.info(f"  Final count: {stats['final_count']}")
    
    return resolved_df, resolution_stats

