# Import required libraries
import logging
from node2vec import Node2Vec
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import community
import networkx as nx
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from recordlinkage import Index, Compare
import recordlinkage
from sklearn.metrics.pairwise import cosine_similarity
from networkx import Graph, connected_components
from typing import Dict, Any, Tuple, List
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import Levenshtein
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # Add this import for progress bars
from concurrent.futures import ThreadPoolExecutor
import os
from kg_build import create_knowledge_graph  # Add this import

# Just get the logger, don't set it up again
logger = logging.getLogger(__name__)

def batch_process_embeddings(embeddings: np.ndarray, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Process embeddings in batches to calculate similarity metrics"""
    n_samples = embeddings.shape[0]
    avg_similarities = np.zeros(n_samples)
    max_similarities = np.zeros(n_samples)
    
    # Normalize embeddings for faster cosine similarity computation
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = normalized_embeddings[i:batch_end]
        
        # Compute similarity matrix for the batch
        similarity_batch = np.dot(batch, normalized_embeddings.T)
        
        # Calculate metrics for the batch
        avg_similarities[i:batch_end] = np.mean(similarity_batch, axis=1)
        max_similarities[i:batch_end] = np.max(similarity_batch, axis=1)
    
    return avg_similarities, max_similarities

def create_node_embeddings(G: nx.Graph) -> Dict[str, Any]:
    """Create node embeddings with optimized parameters for large graphs"""
    logger.info(f"\n{'='*100}")
    logger.info(f"{' NODE EMBEDDINGS ':=^100}")
    logger.info(f"{'='*100}\n")
    
    # Get graph size for parameter optimization
    n_nodes = len(G.nodes())
    n_edges = len(G.edges())
    logger.info(f" Graph size:")
    logger.info(f"  • Nodes: {n_nodes:,}")
    logger.info(f"  • Edges: {n_edges:,}")
    
    # Optimize parameters based on graph size
    params = {
        'dimensions': 128,  # Increased for better representation
        'walk_length': 15,  # Optimized from example
        'num_walks': 40,   # Increased for better coverage
        'workers': min(16, os.cpu_count() or 1),
        'p': 1,  # Return parameter
        'q': 1   # In-out parameter
    }
    
    logger.info("\n⚙️ Selected parameters:")
    for param, value in params.items():
        logger.info(f"  • {param}: {value}")
    
    try:
        start_time = datetime.now()
        logger.info("\n🚀 Starting embedding process...")
        
        # Convert to undirected graph for node2vec
        G_undirected = G.to_undirected()
        
        # Create node2vec with optimized parameters
        node2vec = Node2Vec(
            G_undirected,
            dimensions=params['dimensions'],
            walk_length=params['walk_length'],
            num_walks=params['num_walks'],
            workers=params['workers'],
            p=params['p'],
            q=params['q']
        )
        
        # Train model
        logger.info("\n📈 Training Word2Vec model...")
        model = node2vec.fit(window=10, min_count=1)
        
        # Generate embeddings
        logger.info("\n💫 Generating embeddings...")
        embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
        
        # Get node metadata
        labels = [str(node) for node in G.nodes()]
        types = [G.nodes[node].get('type', 'Unknown') for node in G.nodes()]
        tax_ids = [G.nodes[node].get('tax_id', 'N/A') for node in G.nodes()]
        
        # Detect communities using Louvain method
        communities = community.best_partition(G_undirected)
        community_list = [communities[node] for node in G.nodes()]
        
        total_time = datetime.now() - start_time
        logger.info(f"\n✅ Embedding process completed in {str(total_time).split('.')[0]}")
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'types': types,
            'tax_ids': tax_ids,
            'communities': community_list
        }
        
    except Exception as e:
        logger.error(f"❌ Error in embedding creation: {str(e)}")
        logger.info("���️ Falling back to simplified embeddings...")
        return create_simplified_embeddings(G)

def create_simplified_embeddings(G: nx.Graph) -> Dict[str, Any]:
    """Create simplified embeddings for very large graphs"""
    # Calculate basic node features
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    
    # Create simple feature matrix
    features = []
    for node in G.nodes():
        feature_vector = [
            degrees[node],
            clustering[node],
            1 if G.nodes[node].get('type') == 'Issuer' else 0
        ]
        features.append(feature_vector)
    
    embeddings_matrix = np.array(features)
    
    return {
        'embeddings': embeddings_matrix,
        'labels': [str(node) for node in G.nodes()],
        'types': [G.nodes[node].get('type', 'Unknown') for node in G.nodes()],
        'tax_ids': [G.nodes[node].get('tax_id', 'N/A') for node in G.nodes()],
        'communities': [0] * len(G.nodes())  # Simplified community detection
    }

def visualize_embeddings_3d(embedding_data, output_path):
    """Create and save 3D visualization with attribute information"""
    # Reduce dimensionality to 3D
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embedding_data['embeddings'])
    
    # Create DataFrame with all metadata
    df = pd.DataFrame({
        'PC1': embeddings_3d[:, 0],
        'PC2': embeddings_3d[:, 1],
        'PC3': embeddings_3d[:, 2],
        'Label': embedding_data['labels'],
        'Type': embedding_data['types'],
        'Tax_ID': embedding_data['tax_ids'],
        'Community': embedding_data['communities']
    })
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=df['PC1'],
        y=df['PC2'],
        z=df['PC3'],
        mode='markers',
        marker=dict(
            size=4,
            opacity=0.7,
            color=df['Community'],
            colorscale='Viridis'
        ),
        text=df['Label'],
        hovertemplate="<br>".join([
            "Label: %{text}",
            "Type: %{customdata[0]}",
            "Tax ID: %{customdata[1]}", 
            "Community: %{marker.color}",
            "PC1: %{x:.2f}",
            "PC2: %{y:.2f}",
            "PC3: %{z:.2f}"
        ]),
        customdata=np.stack((df['Type'], df['Tax_ID']), axis=1),
        name='Nodes'
    ))
    
    # Update layout
    fig.update_layout(
        title='3D Node Embedding Visualization (Colored by Community)',
        title_x=0.5,
        scene=dict(
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            zaxis_title='Third Principal Component',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.write_html(output_path, include_plotlyjs='cdn')

def calculate_embedding_statistics(embedding_data: Dict[str, Any], G: nx.Graph) -> pd.DataFrame:
    """Calculate embedding statistics with optimized processing"""
    logger.info(f"\n{'='*100}")
    logger.info(f"{' EMBEDDING STATISTICS ':=^100}")
    
    start_time = datetime.now()
    
    # Create DataFrame with all metadata
    stats_dict = {
        'Node': embedding_data['labels'],
        'Type': embedding_data['types'],
        'Tax_ID': embedding_data['tax_ids'],
        'Community': embedding_data['communities']
    }
    
    # Calculate basic embedding statistics
    embeddings = embedding_data['embeddings']
    stats_dict.update({
        'Embedding_Mean': np.mean(embeddings, axis=1),
        'Embedding_Std': np.std(embeddings, axis=1),
        'Embedding_Min': np.min(embeddings, axis=1),
        'Embedding_Max': np.max(embeddings, axis=1),
        'Embedding_L2_Norm': np.linalg.norm(embeddings, axis=1)
    })
    
    # Calculate community sizes
    community_sizes = pd.Series(embedding_data['communities']).value_counts()
    stats_dict['Community_Size'] = pd.Series(embedding_data['communities']).map(community_sizes).values
    
    # Calculate centrality metrics
    logger.info("\n📈 Computing centrality metrics...")
    simple_G = G.to_undirected()
    
    # Calculate centrality metrics in parallel
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
        futures = {
            'Betweenness_Centrality': executor.submit(nx.betweenness_centrality, simple_G),
            'Eigenvector_Centrality': executor.submit(nx.eigenvector_centrality, simple_G, max_iter=100),
            'PageRank': executor.submit(nx.pagerank, simple_G),
            'Degree_Centrality': executor.submit(nx.degree_centrality, simple_G),
            'Closeness_Centrality': executor.submit(nx.closeness_centrality, simple_G)
        }
        
        for metric, future in futures.items():
            try:
                stats_dict[metric] = list(future.result().values())
                logger.info(f"  ✓ {metric} computed")
            except:
                stats_dict[metric] = list(nx.degree_centrality(simple_G).values())
                logger.warning(f"  ⚠️ Fallback to degree centrality for {metric}")
    
    # Calculate similarity metrics
    logger.info("\n🔄 Computing similarity metrics...")
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    stats_dict['Avg_Similarity_To_Others'] = np.mean(similarity_matrix, axis=1)
    stats_dict['Max_Similarity_To_Others'] = np.max(similarity_matrix, axis=1) - 1  # Subtract self-similarity
    
    # Create final DataFrame
    stats_df = pd.DataFrame(stats_dict)
    
    total_time = datetime.now() - start_time
    logger.info(f"\n✅ Statistics calculation completed in {str(total_time).split('.')[0]}")
    
    return stats_df

def create_subgraphs_by_criteria(data: pd.DataFrame) -> List[Tuple[nx.Graph, Dict]]:
    """Create subgraphs based on criteria with optimized processing"""
    logger.info("\n" + "="*100)
    logger.info(f"{' STARTING SUBGRAPH CREATION ':=^100}")
    
    # Filter data first
    data = data[
        (data['Minőségi kritérium alkalmazásra került'] == 'Igen') |
        ((data['Költség kritérium alkalmazásra került'] == 'Nem') & 
         (data['Ár kritérium alkalmazásra került'] == 'Igen'))
    ]
    
    data = data[
        (data['Szerződés/rész odaítélésre került'] != 'Nem') &
        (data['A beszerzés végleges összértéke pénznem'] == 'HUF')
    ]
    
    # Define grouping columns
    grouping_columns = [
        'Ajánlatkérő szervezet főtevékenysége',
        'Ajánlatkérő szervezet típusa',
        'Szerződés típusa',
        'Eljárásrend',
        'Minőségi kritérium alkalmazásra került',
        'Költség kritérium alkalmazásra került',
        'Ár kritérium alkalmazásra került'
    ]
    
    # Drop rows with NaN values in key columns
    key_columns = [
        'Ajánlatkérő szervezet neve',
        'Nyertes ajánlattevő neve',
        'Teljesítés helye NUTS-kód(ok)',
        'Beérkezett ajánlatok száma',
        'Eljárás EKR azonosító'
    ] + grouping_columns
    
    data = data.dropna(subset=key_columns)
    
    # Create subgraphs
    subgraphs = []
    grouped = data.groupby(grouping_columns)
    
    total_groups = sum(1 for _, group in grouped if len(group) > 200)
    logger.info(f"\nProcessing {total_groups} valid groups")
    
    for name, group in grouped:
        if len(group) > 200:
            try:
                # Create graph using existing function
                G = create_knowledge_graph(group)
                
                # Create criteria dictionary
                criteria = dict(zip(grouping_columns, name))
                
                subgraphs.append((G, criteria))
                
            except Exception as e:
                logger.error(f"Error creating subgraph for group {name}: {str(e)}")
                continue
    
    logger.info(f"\nCreated {len(subgraphs)} subgraphs")
    return subgraphs

def calculate_subgraph_statistics(G: nx.Graph) -> Dict[str, Any]:
    """Calculate statistics for a subgraph"""
    logger.info(f"{' NETWORK STATISTICS ':=^100}")
    stats = {}
    
    # Basic network statistics
    n_nodes = len(G.nodes())
    n_winners = len([n for n, d in G.nodes(data=True) if d['bipartite'] == 0])
    n_buyers = len([n for n, d in G.nodes(data=True) if d['bipartite'] == 1])
    n_edges = len(G.edges())
    
    logger.info("\nNetwork composition:")
    logger.info(f"  • Total nodes: {n_nodes}")
    logger.info(f"  • Winners: {n_winners}")
    logger.info(f"  • Buyers: {n_buyers}")
    logger.info(f"  • Edges: {n_edges}")
    
    # Connected components
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    stats['Number_of_Subgraphs'] = len(subgraphs)
    
    # Network metrics
    stats['Density'] = nx.density(G)
    stats['Assortativity'] = nx.attribute_assortativity_coefficient(G, 'place')
    
    # Calculate statistics for each connected component
    component_stats = []
    for sg in subgraphs:
        if len([n for n, d in sg.nodes(data=True) if d['bipartite'] == 0]) >= 10:
            sg_stats = {
                'nodes': len(sg.nodes()),
                'edges': len(sg.edges()),
                'winners': len([n for n, d in sg.nodes(data=True) if d['bipartite'] == 0]),
                'buyers': len([n for n, d in sg.nodes(data=True) if d['bipartite'] == 1])
            }
            
            # Additional metrics for connected components
            if nx.is_connected(sg):
                sg_stats['avg_path_length'] = nx.average_shortest_path_length(sg)
                sg_stats['diameter'] = nx.diameter(sg)
            
            sg_stats['density'] = nx.density(sg)
            sg_stats['number_of_cliques'] = nx.number_of_cliques(sg)
            sg_stats['largest_clique_size'] = len(max(nx.find_cliques(sg), key=len))
            sg_stats['k_core_number'] = max(nx.core_number(sg).values())
            
            component_stats.append(sg_stats)
    
    stats['Component_Statistics'] = component_stats
    
    logger.info("\nNetwork metrics:")
    logger.info(f"  • Density: {stats['Density']:.4f}")
    logger.info(f"  • Assortativity: {stats['Assortativity']:.4f}")
    logger.info(f"  • Number of components: {stats['Number_of_Subgraphs']}")
    logger.info(f"{'-'*100}\n")
    
    return stats

def clean_filename(text: str) -> str:
    """Clean text to create a valid filename"""
    # Replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    filename = str(text)
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace spaces and dashes with underscores
    filename = filename.replace(' ', '_').replace('-', '_')
    
    # Remove any other non-ASCII characters
    filename = ''.join(c for c in filename if c.isalnum() or c in '_.')
    
    # Limit length and remove trailing underscores
    filename = filename[:50].rstrip('_')
    
    return filename

def save_network_statistics(subgraphs: List[Tuple[nx.Graph, Dict]], timestamp: str) -> pd.DataFrame:
    """Save network statistics incrementally to CSV files and return combined node statistics"""
    logger.info(f"\n{' SAVING NETWORK STATISTICS ':=^100}")
    
    # Create directories if they don't exist
    output_dirs = ['output/node_statistics', 'output/subgraph_statistics']
    for directory in output_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Define output files with clean names
    combined_stats_file = f'output/node_statistics/combined_node_stats_{timestamp}.csv'
    subgraph_stats_file = f'output/subgraph_statistics/subgraph_stats_{timestamp}.csv'
    
    total_subgraphs = len(subgraphs)
    start_time = datetime.now()
    
    # Initialize files with headers
    initial_stats_df = pd.DataFrame(columns=[
        'Node', 'Winner_or_Buyer', 'Tax_ID', 'Community', 'Embedding_Mean',
        'Embedding_Std', 'Embedding_Min', 'Embedding_Max', 'Embedding_L2_Norm',
        'Community_Size', 'Betweenness_Centrality', 'Eigenvector_Centrality',
        'PageRank', 'Avg_Similarity_To_Others', 'Max_Similarity_To_Others'
    ])
    initial_stats_df.to_csv(combined_stats_file, index=False)
    
    initial_subgraph_df = pd.DataFrame(columns=[
        'nodes', 'edges', 'density', 'avg_degree',
        'Ajánlatkérő szervezet főtevékenysége', 'Ajánlatkérő szervezet típusa',
        'Szerződés típusa', 'Eljárásrend', 'Minőségi kritérium alkalmazásra került',
        'Költség kritérium alkalmazásra került', 'Ár kritérium alkalmazásra került'
    ])
    initial_subgraph_df.to_csv(subgraph_stats_file, index=False)
    
    # Process each subgraph
    for idx, (G, criteria) in enumerate(subgraphs, 1):
        try:
            # Create embeddings for this subgraph
            embedding_data = create_node_embeddings(G)
            
            # Calculate statistics
            stats_df = calculate_embedding_statistics(embedding_data, G)
            
            # Rename Type column to Winner_or_Buyer for consistency
            stats_df = stats_df.rename(columns={'Type': 'Winner_or_Buyer'})
            stats_df['Winner_or_Buyer'] = stats_df['Winner_or_Buyer'].map({
                'Issuer': 'Buyer',
                'Winner': 'Winner'
            })
            
            # Add criteria information
            for key, value in criteria.items():
                stats_df[key] = value
            
            # Create subgraph-level statistics
            subgraph_stats = pd.DataFrame([{
                **criteria,
                'nodes': len(G.nodes()),
                'edges': len(G.edges()),
                'density': nx.density(G),
                'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes())
            }])
            
            # Create a clean filename from criteria
            subgraph_name = '_'.join(
                clean_filename(str(v)) 
                for v in criteria.values()
            )
            
            # Create detail file path with clean name
            detail_file = Path('output/node_statistics') / f'subgraph_{idx}_{subgraph_name}_{timestamp}.csv'
            
            # Ensure parent directory exists
            detail_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save files
            stats_df.to_csv(str(detail_file), index=False)
            stats_df.to_csv(combined_stats_file, mode='a', header=False, index=False)
            subgraph_stats.to_csv(subgraph_stats_file, mode='a', header=False, index=False)
            
            # Clear memory
            del stats_df
            del embedding_data
            
        except Exception as e:
            logger.error(f"Error processing subgraph {idx}: {str(e)}")
            logger.exception("Full traceback:")
            continue
        
        # Log memory usage periodically
        if idx % 5 == 0:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    try:
        # Read back the final combined statistics
        combined_node_stats = pd.read_csv(combined_stats_file)
        
        total_time = datetime.now() - start_time
        logger.info(f"\n✅ Statistics saved successfully in {str(total_time).split('.')[0]}")
        logger.info(f"Final files:")
        logger.info(f"  • Combined node statistics: {combined_stats_file}")
        logger.info(f"  • Subgraph statistics: {subgraph_stats_file}")
        
        return combined_node_stats
        
    except Exception as e:
        logger.error(f"Error reading back combined statistics: {str(e)}")
        return pd.DataFrame()

