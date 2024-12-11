import pandas as pd
import networkx as nx
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
from entity_resolution import resolve_entities
from kg_build import create_knowledge_graph
from graph_features import create_subgraphs_by_criteria, calculate_subgraph_statistics, create_node_embeddings, calculate_embedding_statistics, save_network_statistics

# Set up logging once for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger("gensim").setLevel(logging.WARNING)
logging.getLogger("recordlinkage").setLevel(logging.WARNING)

def setup_directories():
    """Create necessary directories for outputs"""
    dirs = ['output', 'output/graphs', 'output/data', 'output/visualizations', 'output/exports']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def clean_data(input_file: str) -> pd.DataFrame:
    """Clean the input data using functions from cleaning.py"""
    logger.info("Starting data cleaning process...")
    logger.info(f"Reading data from {input_file}")
    
    from cleaning import clean_dataset
    
    # Read raw data first to log initial shape
    raw_data = pd.read_csv(input_file)
    logger.info(f"Initial data shape: {raw_data.shape}")
    
    # Clean data with all steps from the notebook
    data = clean_dataset(input_file)
    logger.info(f"Final cleaned data shape: {data.shape}")
    
    return data

def process_subgraphs(data: pd.DataFrame) -> nx.Graph:
    """Process each contract type separately and merge results"""
    from kg_build import create_subgraphs_by_type
    from entity_resolution import resolve_entities_for_subgraph, merge_resolved_subgraphs
    
    # Create subgraphs by contract type
    subgraphs = create_subgraphs_by_type(data)
    
    # Create a single DataFrame to store all resolution results
    all_resolutions = []
    
    # Track overall statistics
    total_stats = {
        'total_nodes_processed': 0,
        'total_nodes_merged': 0,
        'tax_id_matches': 0,
        'name_matches': 0,
        'details_by_type': {}
    }
    
    # Process each subgraph
    resolved_subgraphs = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for contract_type, G in subgraphs.items():
        logger.info(f"\nProcessing {contract_type} subgraph...")
        G.graph['contract_type'] = contract_type
        
        type_stats = {'nodes': len(G.nodes()), 'edges': len(G.edges())}
        logger.info(f"Graph has {type_stats['nodes']} nodes and {type_stats['edges']} edges")
        
        # Create directory for this contract type
        contract_dir = Path(f'output/data/{contract_type}')
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        # Process issuers
        logger.info(f"Resolving {contract_type} issuers...")
        issuer_results = resolve_entities_for_subgraph(G, 'Issuer')
        if not issuer_results.empty:
            issuer_results['contract_type'] = contract_type
            all_resolutions.append(issuer_results)
            
            # Save issuer results
            issuer_file = contract_dir / f'issuer_resolution_{timestamp}.csv'
            issuer_results.to_csv(issuer_file, index=False)
            
            tax_id_matches = len(issuer_results[issuer_results['match_type'] == 'tax_id'])
            name_matches = len(issuer_results[issuer_results['match_type'] == 'name_similarity'])
            logger.info(f"Found {tax_id_matches} tax ID matches and {name_matches} name matches for issuers")
            logger.info(f"Saved issuer resolution results to {issuer_file}")
            
        # Process winners
        logger.info(f"Resolving {contract_type} winners...")
        winner_results = resolve_entities_for_subgraph(G, 'Winner')
        if not winner_results.empty:
            winner_results['contract_type'] = contract_type
            all_resolutions.append(winner_results)
            
            # Save winner results
            winner_file = contract_dir / f'winner_resolution_{timestamp}.csv'
            winner_results.to_csv(winner_file, index=False)
            
            tax_id_matches = len(winner_results[winner_results['match_type'] == 'tax_id'])
            name_matches = len(winner_results[winner_results['match_type'] == 'name_similarity'])
            logger.info(f"Found {tax_id_matches} tax ID matches and {name_matches} name matches for winners")
            logger.info(f"Saved winner resolution results to {winner_file}")
        
        # Save the subgraph
        graph_file = contract_dir / f'graph_{timestamp}.gexf'
        nx.write_gexf(G, graph_file)
        logger.info(f"Saved processed graph to {graph_file}")
        
        resolved_subgraphs[contract_type] = G
        total_stats['total_nodes_processed'] += len(G.nodes())
        total_stats['details_by_type'][contract_type] = type_stats
    
    # Combine all resolutions and save summary report
    if all_resolutions:
        combined_resolutions = pd.concat(all_resolutions, ignore_index=True)
        total_stats['total_nodes_merged'] = len(combined_resolutions)
        
        # Save both summary and detailed results
        summary_file = Path(f'output/data/entity_resolution_summary_{timestamp}.json')
        detailed_file = Path(f'output/data/entity_resolution_details_{timestamp}.csv')
        
        # Save summary statistics
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved entity resolution summary to {summary_file}")
        
        # Save detailed results
        combined_resolutions.to_csv(detailed_file, index=False)
        logger.info(f"Saved detailed resolution results to {detailed_file}")
        
        # Print detailed summary
        logger.info("\n=== Entity Resolution Summary ===")
        logger.info(f"Total nodes processed: {total_stats['total_nodes_processed']}")
        logger.info(f"Total nodes merged: {total_stats['total_nodes_merged']}")
        logger.info(f"Total matches by tax ID: {total_stats['tax_id_matches']}")
        logger.info(f"Total matches by name similarity: {total_stats['name_matches']}")
        
        logger.info("\nBreakdown by contract type:")
        for contract_type, stats in total_stats['details_by_type'].items():
            logger.info(f"\n{contract_type}:")
            logger.info(f"  Nodes: {stats['nodes']}")
            logger.info(f"  Edges: {stats['edges']}")
    
    # Merge resolved subgraphs
    final_graph = merge_resolved_subgraphs(resolved_subgraphs)
    
    return final_graph

def filter_and_analyze(G: nx.Graph, data: pd.DataFrame) -> Dict[str, Any]:
    """Filter data and generate analysis"""
    logger.info(f"Starting filtering and analysis with data shape: {data.shape}")
    
    from filtering import analyze_top_companies, analyze_contract_values
    
    # Execute analysis using notebook functions
    analysis_results = {
        'top_companies': analyze_top_companies(data),
        'contract_values': analyze_contract_values(data)
    }
    
    logger.info("Filtering and analysis completed")
    return analysis_results

def create_visualizations(G: nx.Graph, analysis_results: Dict[str, Any]):
    """Create visualizations using pyvis"""
    logger.info("Creating visualizations...")
    
    from visu import create_network_visualization
    
    # Create main network visualization
    create_network_visualization(
        G, 
        output_path='output/visualizations/main_network.html'
    )
    
    # Create subgraph visualizations based on analysis results
    for key, result in analysis_results['top_companies'].items():
        # Access the graph object from the result dictionary
        if isinstance(result, dict) and 'graph' in result and result['graph'] is not None:
            create_network_visualization(
                result['graph'],
                output_path=result['visu_path']
            )
        else:
            logger.warning(f"Skipping visualization for {key} - no valid graph found")
    
    # Create visualizations for contract value analysis
    for key, result in analysis_results['contract_values'].items():
        if isinstance(result, dict) and 'graph' in result and result['graph'] is not None:
            create_network_visualization(
                result['graph'],
                output_path=result['visu_path']
            )
        else:
            logger.warning(f"Skipping visualization for {key} - no valid graph found")
    
    logger.info("Visualizations created")

def analyze_network_features(data: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Analyze network features and append results to original data"""
    logger.info("\n" + "="*80)
    logger.info("STARTING NETWORK FEATURE ANALYSIS")
    logger.info("="*80 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Create subgraphs
        subgraphs = create_subgraphs_by_criteria(data)
        
        # Save network statistics and get node statistics
        node_stats_df = save_network_statistics(subgraphs, timestamp)
        
        if not node_stats_df.empty:
            # Create mappings for both buyer and winner nodes
            buyer_stats = node_stats_df[node_stats_df['Winner_or_Buyer'] == 'Buyer'].set_index('Node')
            winner_stats = node_stats_df[node_stats_df['Winner_or_Buyer'] == 'Winner'].set_index('Node')
            
            # Add statistics to original data
            data = data.merge(
                buyer_stats,
                left_on='Ajánlatkérő szervezet neve',
                right_index=True,
                how='left',
                suffixes=('', '_buyer')
            )
            
            data = data.merge(
                winner_stats,
                left_on='Nyertes ajánlattevő neve',
                right_index=True,
                how='left',
                suffixes=('', '_winner')
            )
            
            logger.info(f"Added network features to {len(data)} rows")
            
        return data
        
    except Exception as e:
        logger.error(f"Error in network feature analysis: {str(e)}")
        logger.exception("Full traceback:")
        return data

def main():
    """Main execution function"""
    logger.info("Starting pipeline execution...")
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Clean Data
        data = clean_data('ekr_export.csv')
        
        # 2. Resolve Entities
        entity_columns = {
            'issuer': {
                'name': 'Ajánlatkérő szervezet neve',
                'tax_id': 'Ajánlatkérő nemzeti azonosítószáma'
            },
            'winner': {
                'name': 'Nyertes ajánlattevő neve',
                'tax_id': 'Nyertes ajánlattevő adószáma (adóazonosító jele)'
            }
        }
        
        # Before entity resolution, create a copy of the original columns
        data['Ajánlatkérő szervezet neve - eredeti'] = data['Ajánlatkérő szervezet neve']
        data['Nyertes ajánlattevő neve - eredeti'] = data['Nyertes ajánlattevő neve']
        
        resolved_data, resolution_stats = resolve_entities(data, entity_columns)
        
        # Save resolved data and stats
        resolved_data.to_csv(f'output/data/resolved_data_{timestamp}.csv', index=False)
        with open(f'output/data/resolution_stats_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(resolution_stats, f, indent=2, ensure_ascii=False)
        
        # 3. Build Knowledge Graph
        G = create_knowledge_graph(resolved_data)
        
        # 4. Filter and Analyze
        analysis_results = filter_and_analyze(G, resolved_data)
        
        # 5. Create Visualizations
        create_visualizations(G, analysis_results)
        
        # Save final graph
        nx.write_gexf(G, f'output/graphs/final_graph_{timestamp}.gexf')
        
        # 6. Analyze Network Features
        enhanced_data = analyze_network_features(resolved_data, G)
        
        # Save enhanced data
        enhanced_data.to_csv(f'output/data/enhanced_data_{timestamp}.csv', index=False)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 