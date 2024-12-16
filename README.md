# Közbeszerzési Hálózatelemzés

Ez a projekt az Elektronikus Közbeszerzési Rendszer (EKR) adatainak hálózatelemzését végzi, korrupciós mintázatok és kockázatok feltárása céljából. A projekt gráf alapú megközelítést és gépi tanulási technikákat alkalmaz a közbeszerzési kapcsolatok elemzésére.

## A Projekt Célja

A projekt fő célja a közbeszerzési hálózatok strukturális elemzése, amely segíthet:
- Potenciális korrupciós kockázatok azonosításában
- Ajánlatkérők és nyertesek közötti kapcsolati hálók feltérképezésében
- Gyanús mintázatok és anomáliák felderítésében
- Közbeszerzési piac koncentrációjának mérésében

## Elemzési Kategóriák

1. **Teljes Hálózat Elemzése**
   - Teljes hálózati struktúra
   - Kiírói hálózat
   - Nyertesek hálózata

2. **Top 15 Elemzések** (szerződéstípusonként)
   - Árubeszerzés
   - Szolgáltatás megrendelés
   - Építési beruházás
   Minden típusnál a top 15 kiíró és nyertes vizsgálata

3. **Értékhatár Alapú Elemzések**
   - 800M+ építési beruházások
   - 300M alatti építési beruházások
   - 900M+ egyajánlatos szerződések

## Hálózati Metrikák

1. **Alapmetrikák**
   - Csúcsok és élek száma
   - Hálózati sűrűség
   - Klaszterezettség
   - Közösségek száma

2. **Centralitás Mértékek**
   - Közöttiség centralitás
   - PageRank
   - Fokszám centralitás
   - Közelség centralitás
   - Sajátvektor centralitás

3. **További Jellemzők**
   - Hálózati átmérő
   - Átlagos úthossz
   - Tranzitivitás
   - Reciprocitás
   - Asszortativitás

## Kimeneti Fájlok

1. **Metrika Fájlok**
   - Gráf szintű metrikák (JSON)
   - Csúcs szintű metrikák (CSV)
   - Bővített adathalmazok metrikákkal (CSV)
   - Centralitás elemzések (CSV)

2. **Vizualizációk**
   - Típus alapú színezésű hálózatok
   - Közösség alapú színezésű hálózatok
   - Interaktív gráf megjelenítések

## Telepítés és Használat

1. Projekt klónozása:
```bash
git clone https://github.com/tamasmakos/Kmonitor_EKR_KnowledgeGraph.git
```

2. Virtuális környezet létrehozása, majd aktiválása (rendszerfüggő)
```bash
python -m venv virtual
```

3. Függőségek telepítése:
```bash
pip install -r requirements.txt
```

4. Program futtatása:
```bash
python main.py
```

## Kimeneti Könyvtárszerkezet

```
output/
├── data/         # Feldolgozott adatfájlok
├── exports/      # CSV és JSON kimenetek
├── graphs/       # Gráf fájlok
└── visualizations/ # Interaktív vizualizációk