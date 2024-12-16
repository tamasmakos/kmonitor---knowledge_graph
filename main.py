import pandas as pd
import networkx as nx
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
from entity_resolution import resolve_entities
from kg_build import create_knowledge_graph
from graph_features import create_subgraphs_by_criteria, calculate_subgraph_statistics
from filtering import (analyze_top_companies, analyze_contract_values)
from visu import create_network_visualization
import shutil  # Add this import at the top

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
    """Create necessary directories for outputs after cleaning up old data"""
    logger.info("Setting up directories...")
    
    # First, remove the entire output directory if it exists
    output_dir = Path('output')
    if output_dir.exists():
        logger.info("Cleaning up old output directory...")
        try:
            shutil.rmtree(output_dir)
            logger.info("Old output directory removed successfully")
        except Exception as e:
            logger.error(f"Error removing old output directory: {str(e)}")
            logger.exception("Full traceback:")
    
    # Create fresh directories
    dirs = ['output/data', 'output/graphs', 'output/visualizations', 'output/exports']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

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

def filter_and_analyze(G: nx.Graph, data: pd.DataFrame) -> Dict[str, Any]:
    """Filter data and generate analysis"""
    logger.info(f"Starting filtering and analysis with data shape: {data.shape}")
    
    try:
        # Pass the full path including the output directory
        output_base = 'output/exports'
        analysis_results = {
            'top_companies': analyze_top_companies(data, output_base),
            'contract_values': analyze_contract_values(data, output_base)
        }
        
        logger.info("Filtering and analysis completed")
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error in filtering and analysis: {str(e)}")
        logger.exception("Full traceback:")
        return {}

def create_visualizations(G: nx.Graph, analysis_results: Dict[str, Any]):
    """Create visualizations using pyvis"""
    logger.info("Creating visualizations...")
    
    # Create main network visualization
    create_network_visualization(
        G, 
        output_path='output/visualizations/main_network.html'
    )
    
    # Create subgraph visualizations based on analysis results
    if 'top_companies' in analysis_results:
        for key, result in analysis_results['top_companies'].items():
            if isinstance(result, dict) and 'graph' in result:
                create_network_visualization(
                    result['graph'],
                    output_path=f'output/visualizations/{key}.html'
                )
    
    if 'contract_values' in analysis_results:
        for key, result in analysis_results['contract_values'].items():
            if isinstance(result, dict) and 'graph' in result:
                create_network_visualization(
                    result['graph'],
                    output_path=f'output/visualizations/{key}.html'
                )
    
    logger.info("Visualizations created")

def main():
    """Main execution function"""
    logger.info("Starting pipeline execution...")
    
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    try:
        # 1. Clean Data
        data = clean_data('ekr_export.csv')
        data.to_csv(f'output/data/cleaned_data_{timestamp}.csv', index=False)
        
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
        
        # Create backup of original names
        data['Ajánlatkérő szervezet neve - eredeti'] = data['Ajánlatkérő szervezet neve']
        data['Nyertes ajánlattevő neve - eredeti'] = data['Nyertes ajánlattevő neve']
        
        resolved_data, resolution_stats = resolve_entities(
            data, 
            entity_columns,
            similarity_threshold=0.85
        )
        
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
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise

if __name__ == "__main__":
    main() 