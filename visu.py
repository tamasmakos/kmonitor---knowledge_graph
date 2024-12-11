import pandas as pd
from pyvis.network import Network
import logging
import networkx as nx
import os

# Set up logging
logger = logging.getLogger(__name__)

def create_network_visualization(G, output_path: str):
    """Create an interactive network visualization using pyvis"""
    if G is None:
        logger.warning(f"Cannot create visualization for {output_path} - graph is None")
        return
        
    logger.info(f"Creating network visualization: {output_path}")
    
    # If data is a DataFrame, convert it to a graph first
    if isinstance(G, pd.DataFrame):
        G = create_graph_from_data(G)
    
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

    # Calculate edge weight scaling
    edge_values = [data.get('value', 0) for _, _, data in G.edges(data=True)]
    if edge_values:
        max_value = max(edge_values)
        min_value = min(edge_values)
        
        # Function to scale values between 1 and 10
        def scale_value(value):
            if max_value == min_value:
                return 1
            scaled = 1 + 9 * ((value - min_value) / (max_value - min_value))
            return scaled
    
    # Add nodes with detailed hover information
    for node in G.nodes():
        # Convert node to string to ensure compatibility
        node_str = str(node)
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Create hover text based on node type
        if node_type == 'Issuer':
            title = (f"Issuer: {node}<br>"
                    f"Tax ID: {node_data.get('tax_id', 'N/A')}<br>"
                    f"Organization Type: {node_data.get('org_type', 'N/A')}<br>"
                    f"Main Activity: {node_data.get('main_activity', 'N/A')}<br>"
                    f"NUTS Code: {node_data.get('nuts_code', 'N/A')}")
        else:  # Winner
            title = (f"Winner: {node}<br>"
                    f"Tax ID: {node_data.get('tax_id', 'N/A')}<br>"
                    f"KKV: {node_data.get('kkv', 'N/A')}<br>"
                    f"Subcontractors: {node_data.get('subcontractors', 'N/A')}")
        
        color = '#dd4b39' if node_type == 'Issuer' else '#162347'
        net.add_node(node_str, label=node_str, color=color, title=title)

    # Add edges with detailed hover information
    for u, v, data in G.edges(data=True):
        # Convert node IDs to strings
        u_str = str(u)
        v_str = str(v)
        value = data.get('value', 0)
        
        # Scale edge width based on value
        width = scale_value(value)
        
        edge_title = (f"Contract ID: {data.get('contract_id', 'N/A')}<br>"
                     f"Type: {data.get('contract_type', 'N/A')}<br>"
                     f"Value: {value:,.2f} {data.get('currency', 'N/A')}<br>"
                     f"Date: {data.get('date', 'N/A')}<br>"
                     f"Quality Criteria: {data.get('quality_criteria', 'N/A')}<br>"
                     f"Cost Criteria: {data.get('cost_criteria', 'N/A')}<br>"
                     f"Price Criteria: {data.get('price_criteria', 'N/A')}<br>"
                     f"Procedure: {data.get('procedure_type', 'N/A')}<br>"
                     f"EU Funded: {data.get('eu_funded', 'N/A')}")
        
        # Edge color based on value
        if value > 1000000000:  # > 1B
            edge_color = '#FF0000'
        elif value > 100000000:  # > 100M
            edge_color = '#FFA500'
        else:
            edge_color = '#A0A0A0'
        
        net.add_edge(u_str, v_str, 
                    title=edge_title, 
                    color=edge_color,
                    width=width)

    # Configure physics
    net.barnes_hut(
        gravity=-2000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )

    # Set options including arrow settings for directed graph
    net.set_options("""
    var options = {
        "nodes": {
            "font": {
                "size": 12
            }
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
                "type": "continuous",
                "forceDirection": "none"
            },
            "scaling": {
                "min": 1,
                "max": 10
            }
        },
        "physics": {
            "enabled": true,
            "barnesHut": {
                "theta": 0.5,
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.05,
                "damping": 0.09,
                "avoidOverlap": 0
            },
            "maxVelocity": 50,
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }
    """)

    # Save the visualization
    try:
        net.save_graph(output_path)
        logger.info(f"Visualization saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization: {str(e)}")

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
            output_path = f'output/visualizations/{file.replace(".csv", ".html")}'
            create_network_visualization(G, output_path)
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


