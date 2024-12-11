import pandas as pd
import networkx as nx
from typing import Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

from typing import Dict

from typing import Dict

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data before analysis"""
    logger.info(f"Starting preprocessing with shape: {data.shape}")
    
    # Drop rows with missing contract values
    data = data.dropna(subset=['A beszerzés végleges összértéke'])
    logger.info(f"After dropping missing contract values: {data.shape}")
    
    # Check for inconsistent values in EKR IDs
    inconsistent_osszertek = data.groupby('Eljárás EKR azonosító')['A szerződés/rész végleges összértéke'].nunique()
    inconsistent_osszertek = inconsistent_osszertek[inconsistent_osszertek > 1]
    logger.info(f"Found {len(inconsistent_osszertek)} EKR IDs with inconsistent values")
    
    # Remove records with inconsistent values
    data = data[~data['Eljárás EKR azonosító'].isin(inconsistent_osszertek.index)]
    logger.info(f"After removing inconsistent values: {data.shape}")
    
    # NO TEXT CLEANING - we keep all original values
    return data

def analyze_top_companies(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze and export top companies by different criteria"""
    logger.info(f"Starting company analysis with data shape: {data.shape}")
    
    from kg_build import create_knowledge_graph
    
    # Preprocess data
    data = preprocess_data(data)
    results = {}
    
    # For each contract type
    for contract_type in ['Árubeszerzés', 'Szolgáltatás megrendelés', 'Építési beruházás']:
        type_data = data[data['Szerződés típusa'] == contract_type]
        
        # Top 15 issuers - count unique EKR IDs per issuer
        top_15_issuers = (type_data.groupby(['Ajánlatkérő szervezet neve'])
                         .agg({'Eljárás EKR azonosító': 'nunique'})
                         .rename(columns={'Eljárás EKR azonosító': 'count'})
                         .sort_values('count', ascending=False)
                         .head(15))
        
        # Export issuers
        filtered_data = data[
            (data['Ajánlatkérő szervezet neve'].isin(top_15_issuers.index)) & 
            (data['Szerződés típusa'] == contract_type)
        ]
        filtered_data.to_csv(f'output/exports/top_15_{contract_type.lower().replace(" ", "_")}_kiiro.csv', index=False)
        
        # Create graph for issuers
        G_issuers = create_knowledge_graph(filtered_data)
        results[f'top_15_{contract_type}_issuers'] = {
            'data': filtered_data,
            'graph': G_issuers,
            'visu_path': f'output/visualizations/top_15_{contract_type.lower().replace(" ", "_")}_kiiro.html'
        }
        
        # Top 15 winners - count unique EKR IDs per winner
        top_15_winners = (type_data.groupby(['Nyertes ajánlattevő neve'])
                         .agg({'Eljárás EKR azonosító': 'nunique'})
                         .rename(columns={'Eljárás EKR azonosító': 'count'})
                         .sort_values('count', ascending=False)
                         .head(15))
        
        # Export winners
        filtered_data = data[
            (data['Nyertes ajánlattevő neve'].isin(top_15_winners.index)) & 
            (data['Szerződés típusa'] == contract_type)
        ]
        filtered_data.to_csv(f'output/exports/top_15_{contract_type.lower().replace(" ", "_")}_nyertes.csv', index=False)
        
        # Create graph for winners
        G_winners = create_knowledge_graph(filtered_data)
        results[f'top_15_{contract_type}_winners'] = {
            'data': filtered_data,
            'graph': G_winners,
            'visu_path': f'output/visualizations/top_15_{contract_type.lower().replace(" ", "_")}_nyertes.html'
        }
    
    return results

def analyze_contract_values(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze and export contract values by different criteria"""
    from kg_build import create_knowledge_graph
    
    # Preprocess data
    data = preprocess_data(data)
    results = {}
    
    # Construction investments over 800M
    grouped_data_800 = (data[data['Szerződés típusa'] == 'Építési beruházás']
                       .groupby('Eljárás EKR azonosító')['A beszerzés végleges összértéke']
                       .max()
                       .reset_index())
    epitesi_osszertek_800M = grouped_data_800[grouped_data_800['A beszerzés végleges összértéke'] > 800000000]
    
    # Get full data for 800M+ projects
    epitesi_osszertek_800M_full = data[data['Eljárás EKR azonosító'].isin(epitesi_osszertek_800M['Eljárás EKR azonosító'])]
    
    # Top 5 issuers and winners for 800M+ construction
    top_5_kiiro_800 = epitesi_osszertek_800M_full['Ajánlatkérő szervezet neve'].value_counts().head(5)
    top_5_nyertes_800 = epitesi_osszertek_800M_full['Nyertes ajánlattevő neve'].value_counts().head(5)
    
    # Filter and export 800M data
    top_5_kiiro_800_data = data[
        (data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_800.index)) & 
        (data['Szerződés típusa'] == 'Építési beruházás')
    ]
    G_kiiro_800 = create_knowledge_graph(top_5_kiiro_800_data)
    results['top_5_kiiro_800M'] = {
        'data': top_5_kiiro_800_data,
        'graph': G_kiiro_800,
        'visu_path': 'output/visualizations/top_5_kiiro_800M_epitesi.html'
    }
    
    top_5_nyertes_800_data = data[
        (data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_800.index)) & 
        (data['Szerződés típusa'] == 'Építési beruházás')
    ]
    G_nyertes_800 = create_knowledge_graph(top_5_nyertes_800_data)
    results['top_5_nyertes_800M'] = {
        'data': top_5_nyertes_800_data,
        'graph': G_nyertes_800,
        'visu_path': 'output/visualizations/top_5_nyertes_800M_epitesi.html'
    }
    
    print(f"Shape of top 5 issuers 800M+ construction data: {top_5_kiiro_800_data.shape}")
    print(f"Shape of top 5 winners 800M+ construction data: {top_5_nyertes_800_data.shape}")
    
    # Construction investments under 300M
    grouped_data_300 = (data[data['Szerződés típusa'] == 'Építési beruházás']
                       .groupby('Eljárás EKR azonosító')['A beszerzés végleges összértéke']
                       .max()
                       .reset_index())
    epitesi_osszertek_300M = grouped_data_300[grouped_data_300['A beszerzés végleges összértéke'] < 300000000]
    
    # Get full data for under 300M projects
    epitesi_osszertek_300M_full = data[data['Eljárás EKR azonosító'].isin(epitesi_osszertek_300M['Eljárás EKR azonosító'])]
    
    # Top 5 issuers and winners for under 300M construction
    top_5_kiiro_300 = epitesi_osszertek_300M_full['Ajánlatkérő szervezet neve'].value_counts().head(5)
    top_5_nyertes_300 = epitesi_osszertek_300M_full['Nyertes ajánlattevő neve'].value_counts().head(5)
    
    # Filter and export 300M data
    top_5_kiiro_300_data = data[
        (data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_300.index)) & 
        (data['Szerződés típusa'] == 'Építési beruházás')
    ]
    top_5_nyertes_300_data = data[
        (data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_300.index)) & 
        (data['Szerződés típusa'] == 'Építési beruházás')
    ]
    
    top_5_kiiro_300_data.to_csv('output/exports/top_5_kiiro_300M_epitesi.csv', index=False)
    top_5_nyertes_300_data.to_csv('output/exports/top_5_nyertes_300M_epitesi.csv', index=False)
    print(f"Shape of top 5 issuers <300M construction data: {top_5_kiiro_300_data.shape}")
    print(f"Shape of top 5 winners <300M construction data: {top_5_nyertes_300_data.shape}")
    
    # For 300M data, create graphs and add them to results
    G_kiiro_300 = create_knowledge_graph(top_5_kiiro_300_data)
    G_nyertes_300 = create_knowledge_graph(top_5_nyertes_300_data)
    
    results['top_5_kiiro_300M'] = {
        'data': top_5_kiiro_300_data,
        'graph': G_kiiro_300,
        'visu_path': 'output/visualizations/top_5_kiiro_300M_epitesi.html'
    }
    
    results['top_5_nyertes_300M'] = {
        'data': top_5_nyertes_300_data,
        'graph': G_nyertes_300,
        'visu_path': 'output/visualizations/top_5_nyertes_300M_epitesi.html'
    }
    
    # Single bidder contracts over 900M
    grouped_data_900_egy = (data[data['Beérkezett ajánlatok száma'] == 1.0]
                           .groupby('Eljárás EKR azonosító')['A beszerzés végleges összértéke']
                           .max()
                           .reset_index())
    osszertek_900M_egy_ajanlat = grouped_data_900_egy[grouped_data_900_egy['A beszerzés végleges összértéke'] > 900000000]
    
    # Get full data for 900M+ single bidder
    osszertek_900M_egy_ajanlat_full = data[data['Eljárás EKR azonosító'].isin(osszertek_900M_egy_ajanlat['Eljárás EKR azonosító'])]
    
    # Top 5 issuers and winners for 900M+ single bidder
    top_5_kiiro_900M_egy = osszertek_900M_egy_ajanlat_full['Ajánlatkérő szervezet neve'].value_counts().head(5)
    top_5_nyertes_900M_egy = osszertek_900M_egy_ajanlat_full['Nyertes ajánlattevő neve'].value_counts().head(5)
    
    # Filter and export 900M single bidder data
    top_5_kiiro_900M_egy_data = data[data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_900M_egy.index)]
    top_5_nyertes_900M_egy_data = data[data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_900M_egy.index)]
    
    top_5_kiiro_900M_egy_data.to_csv('output/exports/top_5_kiiro_900M_egy_ajanlat.csv', index=False)
    top_5_nyertes_900M_egy_data.to_csv('output/exports/top_5_nyertes_900M_egy_ajanlat.csv', index=False)
    print(f"Shape of top 5 issuers 900M+ single bidder data: {top_5_kiiro_900M_egy_data.shape}")
    print(f"Shape of top 5 winners 900M+ single bidder data: {top_5_nyertes_900M_egy_data.shape}")
    
    # Create graphs for 900M single bidder data
    G_kiiro_900M_egy = create_knowledge_graph(top_5_kiiro_900M_egy_data)
    G_nyertes_900M_egy = create_knowledge_graph(top_5_nyertes_900M_egy_data)
    
    results['top_5_kiiro_900M_egy_ajanlat'] = {
        'data': top_5_kiiro_900M_egy_data,
        'graph': G_kiiro_900M_egy,
        'visu_path': 'output/visualizations/top_5_kiiro_900M_egy_ajanlat.html'
    }
    
    results['top_5_nyertes_900M_egy_ajanlat'] = {
        'data': top_5_nyertes_900M_egy_data,
        'graph': G_nyertes_900M_egy,
        'visu_path': 'output/visualizations/top_5_nyertes_900M_egy_ajanlat.html'
    }
    
    return results

def generate_statistics(G: nx.Graph, data: pd.DataFrame) -> Dict[str, Any]:
    """Generate statistics about the network and contracts"""
    logger.info("Generating statistics...")
    
    # Preprocess data
    data = preprocess_data(data)
    
    stats = {
        'graph_stats': {
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / len(G)
        },
        'contract_stats': {
            'total_contracts': len(data),
            'avg_value': data['A szerződés/rész végleges összértéke'].mean(),
            'median_value': data['A szerződés/rész végleges összértéke'].median(),
            'total_value': data['A szerződés/rész végleges összértéke'].sum(),
            'contracts_over_800M': len(data[data['A szerződés/rész végleges összértéke'] > 800000000]),
            'single_bidder_contracts': len(data[data['Beérkezett ajánlatok száma'] == 1])
        }
    }
    
    logger.info("Statistics generation completed")
    return stats

