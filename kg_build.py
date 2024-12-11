import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def create_knowledge_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Create knowledge graph from resolved data"""
    logger.info("Creating knowledge graph from resolved data...")
    
    G = nx.DiGraph()
    
    # Add issuer nodes
    for _, row in df.iterrows():
        issuer_name = row['Ajánlatkérő szervezet neve']
        issuer_id = row['Ajánlatkérő nemzeti azonosítószáma']
        
        if not G.has_node(issuer_name):
            G.add_node(issuer_name, 
                      type='Issuer',
                      tax_id=str(issuer_id),
                      org_type=str(row['Ajánlatkérő szervezet típusa']),
                      main_activity=str(row['Ajánlatkérő szervezet főtevékenysége']),
                      nuts_code=str(row['Teljesítés helye NUTS-kód(ok)']))
        
        # Add winner nodes
        winner_name = row['Nyertes ajánlattevő neve']
        winner_id = row['Nyertes ajánlattevő adószáma (adóazonosító jele)']
        
        if not G.has_node(winner_name):
            G.add_node(winner_name,
                      type='Winner',
                      tax_id=str(winner_id),
                      kkv=str(row['Nyertes ajánlattevő kkv']),
                      subcontractors=str(row['Alvállalkozók igénybevétele a szerződés teljesítéséhez']))
        
        # Add directed edge representing the contract (from issuer to winner)
        G.add_edge(issuer_name, winner_name,
                  # Contract identification
                  contract_id=str(row['Eljárás EKR azonosító']),
                  contract_type=str(row['Szerződés típusa']),
                  
                  # Financial information
                  value=float(row['A szerződés/rész végleges összértéke']) if pd.notna(row['A szerződés/rész végleges összértéke']) else 0.0,
                  currency=str(row['A szerződés/rész végleges összértéke pénznem ']),
                  
                  # Dates
                  date=row['Szerződés megkötésének dátuma'].strftime('%Y-%m-%d') if pd.notna(row['Szerződés megkötésének dátuma']) else '',
                  
                  # Criteria information
                  quality_criteria=str(row['Minőségi kritérium alkalmazásra került']),
                  cost_criteria=str(row['Költség kritérium alkalmazásra került']),
                  price_criteria=str(row['Ár kritérium alkalmazásra került']),
                  
                  # Procedure information
                  procedure_type=str(row['Eljárásrend']),
                  
                  # EU funding
                  eu_funded=str(row['A beszerzés európai uniós alapokból finanszírozott projekttel és/vagy programmal kapcsolatos']))
    
    logger.info(f"Created directed graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G


