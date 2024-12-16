import pandas as pd
from pyvis.network import Network
import logging
import networkx as nx
import os
from networkx.algorithms import community
import random
import re

# Set up logging
logger = logging.getLogger(__name__)

def truncate_name(name: str, max_length: int = 30) -> str:
    """Truncate name if it exceeds max_length characters"""
    return name if len(name) <= max_length else name[:max_length-3] + '...'

def create_network_visualization(G, output_path: str):
    """Create two interactive network visualizations using pyvis - one with type colors, one with community colors"""
    if G is None:
        logger.warning(f"Cannot create visualization for {output_path} - graph is None")
        return
        
    logger.info(f"Creating network visualizations: {output_path}")
    
    # Ensure output path is in the visualizations directory and remove timestamp if present
    output_path = output_path.replace('output/exports', 'output/visualizations')
    output_path = output_path.replace('.html', '')  # Remove .html extension
    # Remove any timestamp pattern like _YYYYMMDD_HHMM
    output_path = re.sub(r'_\d{8}_\d{4}', '', output_path)
    
    # Detect communities
    communities = list(community.louvain_communities(G.to_undirected()))
    
    # Generate colors for communities
    community_colors = {}
    for i in range(len(communities)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        community_colors[i] = f'#{r:02x}{g:02x}{b:02x}'
    
    # Create two visualizations with clean filenames
    for viz_type in ['type_colors', 'community_colors']:
        # Create a pyvis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#FFFFFF",
            font_color="black",
            directed=True,
            notebook=False,
            cdn_resources='remote'
        )

        # Add nodes with appropriate coloring
        for node in G.nodes():
            node_str = str(node)
            truncated_label = truncate_name(node_str)
            
            if viz_type == 'type_colors':
                # Original coloring by node type
                node_type = G.nodes[node].get('type', 'Unknown')
                color = '#dd4b39' if node_type == 'Issuer' else '#162347'
            else:
                # Community-based coloring
                node_community = None
                for i, comm in enumerate(communities):
                    if node in comm:
                        node_community = i
                        break
                color = community_colors.get(node_community, '#808080')
            
            net.add_node(
                node_str,
                label=truncated_label,
                title=node_str,
                color=color
            )

        # Add edges
        for u, v, data in G.edges(data=True):
            u_str = str(u)
            v_str = str(v)
            value = data.get('value', 0)
            
            # Scale edge width based on value
            width = 1 + min(9, value / 100000000)
            
            edge_title = f"Contract value: {value:,.0f} {data.get('currency', 'HUF')}"
            
            # Edge color based on value
            if value > 1000000000:  # > 1B
                edge_color = '#FF0000'
            elif value > 100000000:  # > 100M
                edge_color = '#FFA500'
            else:
                edge_color = '#A0A0A0'
            
            net.add_edge(
                u_str,
                v_str,
                title=edge_title,
                color=edge_color,
                width=width
            )

        # Configure physics
        net.barnes_hut(
            gravity=-2000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09
        )

        # Set options
        net.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 14,
                    "face": "arial"
                },
                "shape": "dot",
                "size": 20
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "color": {
                    "inherit": false
                },
                "smooth": {
                    "type": "continuous"
                }
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 200,
                    "springConstant": 0.05,
                    "damping": 0.09
                },
                "minVelocity": 0.75
            }
        }
        """)

        # Save with clean filename
        output_path_with_suffix = f"{output_path}_{viz_type}.html"
        try:
            net.save_graph(output_path_with_suffix)
            logger.info(f"Saved {viz_type} visualization to {output_path_with_suffix}")
        except Exception as e:
            logger.error(f"Failed to save {viz_type} visualization: {str(e)}")

def create_visualizations_for_exports():
    """Create visualizations for all exported datasets"""
    logger.info("Creating visualizations for exported datasets...")
    
    # Get all CSV files from exports directory
    export_path = 'output/exports'
    export_files = [f for f in os.listdir(export_path) if f.endswith('.csv')]
    
    for file in export_files:
        try:
            data = pd.read_csv(f'output/exports/{file}')
            G = create_graph_from_data(data)
            
            # Create visualization without communities
            output_path = f'output/visualizations/{file.replace(".csv", ".html")}'
            create_network_visualization(G, output_path)
            
            # Detect communities and create visualization with community colors
            communities = list(community.louvain_communities(G))
            output_path_comm = f'output/visualizations/{file.replace(".csv", "_communities.html")}'
            create_network_visualization(G, output_path_comm, communities=communities)
            
        except Exception as e:
            logger.error(f"Failed to create visualization for {file}: {str(e)}")

def create_graph_from_data(data: pd.DataFrame) -> nx.Graph:
    """Create a network graph from the filtered data"""
    G = nx.Graph()
    
    # Add nodes and edges
    for _, row in data.iterrows():
        issuer = row['Ajánlatkérő szervezet neve']
        winner = row['Nyertes ajánlattevő neve']
        
        G.add_node(issuer, type='Issuer')
        G.add_node(winner, type='Winner')
        G.add_edge(issuer, winner)
    
    return G

