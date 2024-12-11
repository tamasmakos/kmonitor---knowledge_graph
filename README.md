# Közbeszerzési Hálózatelemzés

Ez a projekt az Elektronikus Közbeszerzési Rendszer (EKR) adatainak hálózatelemzését végzi, korrupciós mintázatok és kockázatok feltárása céljából. A projekt gráf alapú megközelítést és gépi tanulási technikákat alkalmaz a közbeszerzési kapcsolatok elemzésére.

## A Projekt Célja

A projekt fő célja a közbeszerzési hálózatok strukturális elemzése, amely segíthet:
- Potenciális korrupciós kockázatok azonosításában
- Ajánlatkérők és nyertesek közötti kapcsolati hálók feltérképezésében
- Gyanús mintázatok és anomáliák felderítésében
- Közbeszerzési piac koncentrációjának mérésében

## Adatforrás

Az elemzés az Elektronikus Közbeszerzési Rendszer (EKR) nyilvános adatain alapul. Az adatbázis tartalmazza:
- Ajánlatkérő szervezetek adatait
- Nyertes ajánlattevők információit
- Szerződések részleteit
- Eljárások típusait és értékeit
- Minőségi, költség és ár kritériumokat

## Adatfeldolgozási Pipeline

1. **Adattisztítás** (`cleaning.py`)
   - Hiányzó értékek kezelése
   - Csak odaítélt szerződések megtartása
   - Pénznem szűrése (HUF)
   - Dátumok feldolgozása

2. **Entitás Feloldás** (`entity_resolution.py`)
   - Cégnevek és adószámok egyeztetése
   - Duplikátumok kezelése

3. **Tudásgráf Építés** (`kg_build.py`)
   - Kétoldalú gráf létrehozása (ajánlatkérők és nyertesek)
   - Élek súlyozása szerződési értékekkel
   - Node attribútumok hozzáadása

4. **Hálózati Jellemzők** (`graph_features.py`)

### Hálózatelemzési Szintek:

#### a) Hálózat Szintű Metrikák
- Csúcsok és élek száma
- Hálózati sűrűség
- Asszortativitás

#### b) Komponens Szintű Jellemzők
- Átlagos úthossz
- Átmérő
- Klikkek száma és mérete
- K-mag elemzés

#### c) Csúcs Szintű Metrikák
- Centralitás mértékek (fokszám, közöttiség, közelség)
- Sajátvektor centralitás és PageRank
- Klaszterezési együttható

#### d) Beágyazások és Közösségek
- Node2Vec beágyazások (64 dimenzió)
- Közösségdetektálás
- Hasonlósági mértékek

## Kimeneti Fájlok

A pipeline három fő statisztikai fájlt generál:
1. `subnetwork_stats.csv`: Hálózat szintű statisztikák
2. `subgraph_stats.csv`: Komponens szintű elemzések
3. `node_stats.csv`: Csúcs szintű metrikák és beágyazások


## Használat
Terminal
1, git clone https://github.com/tamasmakos/Kmonitor_EKR_KnowledgeGraph.git
2, python -m virtual venv
3, python -m pip install -r requirements.txt
4, python main.py