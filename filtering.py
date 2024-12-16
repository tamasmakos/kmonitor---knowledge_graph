import pandas as pd
import networkx as nx
from typing import Dict, Any, Tuple
import logging
from graph_features import calculate_embedding_statistics, create_node_embeddings
from kg_build import create_knowledge_graph
import numpy as np
from visu import create_network_visualization
from datetime import datetime
import json

# Set up logging
logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data before analysis"""
    logger.info(f"Starting preprocessing with shape: {data.shape}")
    
    # Instead of dropping, fill missing contract values with 0 or flag them
    data['contract_value_missing'] = data['A beszerzés végleges összértéke'].isna()
    
    # For inconsistent values, instead of removing all records:
    # 1. Group by EKR ID and take the most recent value
    # 2. Or take the average/median of the values
    grouped = data.groupby('Eljárás EKR azonosító')
    
    inconsistent_ids = grouped['A szerződés/rész végleges összértéke'].nunique()
    logger.info(f"Found {len(inconsistent_ids[inconsistent_ids > 1])} EKR IDs with multiple values")
    
    # Take the most recent value or the maximum value
    data = data.sort_values('Hirdetmény közzétételének dátuma', ascending=False)
    data = data.drop_duplicates(subset=['Eljárás EKR azonosító'], keep='first')
    
    logger.info(f"Final shape after deduplication: {data.shape}")
    
    return data

def analyze_graph_and_add_metrics(data: pd.DataFrame, output_prefix: str) -> Tuple[pd.DataFrame, Dict]:
    """Create graph from filtered data, calculate metrics, and save results"""
    logger.info(f"Analyzing graph for {output_prefix}")
    
    # Create graph
    G = create_knowledge_graph(data)
    
    # Calculate embeddings and statistics
    embedding_data = create_node_embeddings(G)
    stats_df = calculate_embedding_statistics(embedding_data, G)
    
    # Enhance data with network metrics
    enhanced_data = data.copy()
    
    # Add issuer metrics
    issuer_stats = stats_df[stats_df['Type'] == 'Issuer'].set_index('Node')
    enhanced_data = enhanced_data.merge(
        issuer_stats[['Betweenness_Centrality', 'PageRank', 'Community', 
                     'Degree_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality']],
        left_on='Ajánlatkérő szervezet neve',
        right_index=True,
        how='left',
        suffixes=('', '_issuer')
    )
    
    # Add winner metrics
    winner_stats = stats_df[stats_df['Type'] == 'Winner'].set_index('Node')
    enhanced_data = enhanced_data.merge(
        winner_stats[['Betweenness_Centrality', 'PageRank', 'Community',
                     'Degree_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality']],
        left_on='Nyertes ajánlattevő neve',
        right_index=True,
        how='left',
        suffixes=('_issuer', '_winner')
    )
    
    # Save enhanced data with metrics
    enhanced_data.to_csv(f'{output_prefix}_with_metrics.csv', index=False)
    
    # Save node-level metrics separately
    stats_df.to_csv(f'{output_prefix}_node_metrics.csv', index=False)
    
    # Create flattened metrics dictionary
    graph_metrics = {
        'analysis_type': output_prefix,
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'number_of_communities': len(set(stats_df['Community'])),
        'avg_betweenness': stats_df['Betweenness_Centrality'].mean(),
        'avg_pagerank': stats_df['PageRank'].mean(),
        'avg_degree_centrality': np.mean(list(nx.degree_centrality(G).values())),
        'avg_closeness_centrality': np.mean(list(nx.closeness_centrality(G).values())),
        'avg_eigenvector_centrality': np.mean(list(nx.eigenvector_centrality(G, max_iter=1000).values())),
        'diameter': nx.diameter(G.to_undirected()) if nx.is_connected(G.to_undirected()) else -1,
        'avg_shortest_path': nx.average_shortest_path_length(G.to_undirected()) if nx.is_connected(G.to_undirected()) else -1,
        'transitivity': nx.transitivity(G),
        'reciprocity': nx.reciprocity(G),
        'assortativity': nx.degree_assortativity_coefficient(G)
    }
    
    # Save metrics as JSON
    with open(f'{output_prefix}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(graph_metrics, f, indent=2, ensure_ascii=False)
    
    return enhanced_data, graph_metrics

def analyze_top_companies(data: pd.DataFrame, output_base: str) -> Dict[str, Any]:
    """Analyze and export top companies by different criteria"""
    logger.info(f"Starting company analysis with shape: {data.shape}")
    
    # Preprocess data
    data = preprocess_data(data)
    results = {}
    graph_metrics = []
    
    # For each contract type
    for contract_type in ['Árubeszerzés', 'Szolgáltatás megrendelés', 'Építési beruházás']:
        type_data = data[data['Szerződés típusa'] == contract_type]
        
        # Top 15 issuers processing
        top_15_issuers = (type_data.groupby(['Ajánlatkérő szervezet neve'])
                         .agg({'Eljárás EKR azonosító': 'nunique'})
                         .rename(columns={'Eljárás EKR azonosító': 'count'})
                         .sort_values('count', ascending=False)
                         .head(15))
        
        filtered_data = data[
            (data['Ajánlatkérő szervezet neve'].isin(top_15_issuers.index)) & 
            (data['Szerződés típusa'] == contract_type)
        ]
        
        # Analyze and save with metrics - use output_base in paths
        output_path = f'{output_base}/top_15_{contract_type.lower().replace(" ", "_")}_kiiro'
        enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_data, output_path)
        metrics['analysis_type'] = f'top_15_{contract_type}_issuers'
        graph_metrics.append(metrics)
        
        # Create visualization for issuers
        G_issuers = create_knowledge_graph(filtered_data)
        visu_path = f'output/visualizations/top_15_{contract_type.lower().replace(" ", "_")}_kiiro.html'
        create_network_visualization(G_issuers, visu_path)
        
        # Similar process for winners
        top_15_winners = (type_data.groupby(['Nyertes ajánlattevő neve'])
                         .agg({'Eljárás EKR azonosító': 'nunique'})
                         .rename(columns={'Eljárás EKR azonosító': 'count'})
                         .sort_values('count', ascending=False)
                         .head(15))
        
        filtered_data = data[
            (data['Nyertes ajánlattevő neve'].isin(top_15_winners.index)) & 
            (data['Szerződés típusa'] == contract_type)
        ]
        
        # Analyze and save with metrics - use output_base in paths
        output_path = f'{output_base}/top_15_{contract_type.lower().replace(" ", "_")}_nyertes'
        enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_data, output_path)
        metrics['analysis_type'] = f'top_15_{contract_type}_winners'
        graph_metrics.append(metrics)
        
        # Create visualization for winners
        G_winners = create_knowledge_graph(filtered_data)
        visu_path = f'output/visualizations/top_15_{contract_type.lower().replace(" ", "_")}_nyertes.html'
        create_network_visualization(G_winners, visu_path)
    
    # Save combined metrics summary
    pd.DataFrame(graph_metrics).to_csv(f'{output_base}/top_companies_graph_metrics.csv', index=False)
    
    return results

def analyze_contract_values(data: pd.DataFrame, output_base: str) -> Dict[str, Any]:
    """Analyze and export contract values by different criteria"""
    # Preprocess data
    data = preprocess_data(data)
    results = {}
    graph_metrics = []
    
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
    
    # Process 800M+ issuers
    filtered_kiiro = data[data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_800.index)]
    output_path = f'{output_base}/top_5_kiiro_800M_epitesi'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_kiiro, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 800M+ issuers
    G_kiiro_800 = create_knowledge_graph(filtered_kiiro)
    create_network_visualization(G_kiiro_800, f'output/visualizations/top_5_kiiro_800M_epitesi.html')
    
    # Process 800M+ winners
    filtered_nyertes = data[data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_800.index)]
    output_path = f'{output_base}/top_5_nyertes_800M_epitesi'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_nyertes, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 800M+ winners
    G_nyertes_800 = create_knowledge_graph(filtered_nyertes)
    create_network_visualization(G_nyertes_800, f'output/visualizations/top_5_nyertes_800M_epitesi.html')
    
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
    
    # Process 300M issuers
    filtered_kiiro = data[data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_300.index)]
    output_path = f'{output_base}/top_5_kiiro_300M_epitesi'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_kiiro, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 300M issuers
    G_kiiro_300 = create_knowledge_graph(filtered_kiiro)
    create_network_visualization(G_kiiro_300, f'output/visualizations/top_5_kiiro_300M_epitesi.html')
    
    # Process 300M winners
    filtered_nyertes = data[data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_300.index)]
    output_path = f'{output_base}/top_5_nyertes_300M_epitesi'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_nyertes, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 300M winners
    G_nyertes_300 = create_knowledge_graph(filtered_nyertes)
    create_network_visualization(G_nyertes_300, f'output/visualizations/top_5_nyertes_300M_epitesi.html')
    
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
    
    # Process 900M single bidder issuers
    filtered_kiiro = data[data['Ajánlatkérő szervezet neve'].isin(top_5_kiiro_900M_egy.index)]
    output_path = f'{output_base}/top_5_kiiro_900M_egy_ajanlat'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_kiiro, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 900M single bidder issuers
    G_kiiro_900 = create_knowledge_graph(filtered_kiiro)
    create_network_visualization(G_kiiro_900, f'output/visualizations/top_5_kiiro_900M_egy_ajanlat.html')
    
    # Process 900M single bidder winners
    filtered_nyertes = data[data['Nyertes ajánlattevő neve'].isin(top_5_nyertes_900M_egy.index)]
    output_path = f'{output_base}/top_5_nyertes_900M_egy_ajanlat'
    enhanced_data, metrics = analyze_graph_and_add_metrics(filtered_nyertes, output_path)
    graph_metrics.append(metrics)
    
    # Create visualization for 900M single bidder winners
    G_nyertes_900 = create_knowledge_graph(filtered_nyertes)
    create_network_visualization(G_nyertes_900, f'output/visualizations/top_5_nyertes_900M_egy_ajanlat.html')
    
    # Save combined metrics
    pd.DataFrame(graph_metrics).to_csv(f'{output_base}/contract_values_metrics.csv', index=False)
    
    return results

def analyze_full_graph(data: pd.DataFrame, output_base: str) -> Dict[str, Any]:
    """Analyze the full graph without any filters"""
    logger.info("Starting full graph analysis...")
    
    # Create the full graph
    G = create_knowledge_graph(data)
    
    # Calculate embeddings and statistics for the full graph
    embedding_data = create_node_embeddings(G)
    stats_df = calculate_embedding_statistics(embedding_data, G)
    
    # Save the full graph statistics
    output_path = f'{output_base}/full_graph'
    enhanced_data, metrics = analyze_graph_and_add_metrics(data, output_path)
    
    # Create visualization for the full graph
    visu_path = 'output/visualizations/full_graph.html'
    create_network_visualization(G, visu_path)
    
    return {
        'data': enhanced_data,
        'metrics': metrics,
        'graph': G
    }

def create_combined_centrality_analysis():
    """Create a combined analysis of centrality measures from all exported CSVs"""
    import glob
    
    # Get all exported CSVs and their corresponding metrics
    csv_files = glob.glob('output/exports/*.csv')
    metrics_files = glob.glob('output/exports/*_metrics.csv')  # Changed from *_graph_metrics.json
    
    # Initialize DataFrames
    detailed_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    
    try:
        # Combine all metrics into one summary
        all_metrics = []
        for metrics_file in metrics_files:
            if 'centrality_analysis' in metrics_file:  # Skip the summary file itself
                continue
            
            metrics = pd.read_csv(metrics_file)  # Changed from read_json
            analysis_type = metrics_file.split('/')[-1].replace('_metrics.csv', '')
            metrics['analysis_type'] = analysis_type
            all_metrics.append(metrics)
        
        if all_metrics:
            # Create summary DataFrame
            summary_df = pd.concat(all_metrics, ignore_index=True)
            summary_df.to_csv('output/exports/centrality_analysis_summary.csv', index=False)
            
            # Create detailed DataFrame by combining all CSVs
            detailed_dfs = []
            for csv_file in csv_files:
                if '_metrics.csv' in csv_file or 'centrality_analysis' in csv_file:
                    continue
                    
                df = pd.read_csv(csv_file)
                analysis_type = csv_file.split('/')[-1].replace('.csv', '')
                df['analysis_type'] = analysis_type
                detailed_dfs.append(df)
            
            if detailed_dfs:
                detailed_df = pd.concat(detailed_dfs, ignore_index=True)
                detailed_df.to_csv('output/exports/centrality_analysis_detailed.csv', index=False)
        
        logger.info(f"Created summary with {len(summary_df)} entries and detailed analysis with {len(detailed_df)} entries")
        
    except Exception as e:
        logger.error(f"Error in creating centrality analysis: {str(e)}")
        logger.exception("Full traceback:")
    
    return detailed_df, summary_df

def filter_and_analyze(data: pd.DataFrame, output_base: str) -> Dict[str, Any]:
    """Main function to run all analyses"""
    logger.info("Starting comprehensive analysis...")
    
    # First, analyze the full network without any filters
    logger.info("Analyzing full network...")
    G_full = create_knowledge_graph(data)
    output_path = f'{output_base}/full_network'
    enhanced_data, full_metrics = analyze_graph_and_add_metrics(data, output_path)
    
    # Save full network metrics separately
    full_metrics['analysis_type'] = 'full_network'
    pd.DataFrame([full_metrics]).to_csv(f'{output_base}/full_network_metrics.csv', index=False)
    
    # Create visualization for full network
    create_network_visualization(G_full, f'output/visualizations/full_network.html')
    
    # Analyze total network (new addition)
    total_network_results = analyze_total_network(data, output_base)
    
    # Continue with other analyses
    results = {
        'full_network': {
            'data': enhanced_data,
            'metrics': full_metrics,
            'graph': G_full
        },
        'total_network': total_network_results,
        'top_companies': analyze_top_companies(data, output_base),
        'contract_values': analyze_contract_values(data, output_base)
    }
    
    return results

def analyze_total_network(data: pd.DataFrame, output_base: str) -> Dict[str, Any]:
    """Analyze the complete network without any filters"""
    logger.info("Starting total network analysis...")
    
    # Preprocess data
    data = preprocess_data(data)
    results = {}
    graph_metrics = []
    
    # Create total network graph
    G_total = create_knowledge_graph(data)
    
    # 1. All Issuers Network
    issuer_data = data.copy()
    output_path = f'{output_base}/total_network_issuers'
    enhanced_data, metrics = analyze_graph_and_add_metrics(issuer_data, output_path)
    metrics['analysis_type'] = 'total_network_issuers'
    graph_metrics.append(metrics)
    
    # Create visualization for all issuers
    visu_path = f'output/visualizations/total_network_issuers.html'
    create_network_visualization(G_total, visu_path)
    
    # 2. All Winners Network
    winner_data = data.copy()
    output_path = f'{output_base}/total_network_winners'
    enhanced_data, metrics = analyze_graph_and_add_metrics(winner_data, output_path)
    metrics['analysis_type'] = 'total_network_winners'
    graph_metrics.append(metrics)
    
    # Create visualization for all winners
    visu_path = f'output/visualizations/total_network_winners.html'
    create_network_visualization(G_total, visu_path)
    
    # 3. Complete Network
    output_path = f'{output_base}/total_network_complete'
    enhanced_data, metrics = analyze_graph_and_add_metrics(data, output_path)
    metrics['analysis_type'] = 'total_network_complete'
    graph_metrics.append(metrics)
    
    # Create visualization for complete network
    visu_path = f'output/visualizations/total_network_complete.html'
    create_network_visualization(G_total, visu_path)
    
    # Save combined metrics summary
    pd.DataFrame(graph_metrics).to_csv(f'{output_base}/total_network_metrics.csv', index=False)
    
    return results

