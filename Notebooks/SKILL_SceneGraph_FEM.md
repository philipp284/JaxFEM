# Skill: berechne_SceneGraph / IFC-FEM it

**Trigger:** Philipp sagt `berechne_SceneGraph`, `scene-graph-fem`, `IFC-FEM it`, oder sinngemäß  
„analysiere dieses IFC-Gebäude mit dem Scene Graph" / „führe eine FEM auf dieser IFC-Datei durch".

---

## Was dieser Skill tut

Claude agiert als Berechnungsingenieur und führt eine vollautomatische strukturmechanische FEM-Berechnung auf einem IFC-Gebäudemodell durch. Grundlage ist das Notebook `IFC_SceneGraph_FEM.ipynb`.

---

## Ablauf (Schritt für Schritt)

### 1. Parameteraufnahme

Philipp gibt an (oder Claude fragt nach):

| Parameter | Beschreibung | Standard |
|-----------|-------------|---------|
| `IFC_PATH` | Pfad zur IFC-Datei | Duplex-Demo-Modell |
| `MESH_SIZE` | Elementgröße [m] | `1.0` (Debug) / `0.3` (Produktion) |
| `Q_NUTZLAST` | Nutzlast [kN/m²] | `5.0` |
| `E_DEFAULT` | E-Modul Fallback [kN/m²] | `30 000 000` (C25/30) |
| `NU_DEFAULT` | Querdehnzahl Fallback | `0.2` |
| `Ausgabename` | Bezeichnung für Dateien | aus IFC-Dateiname |

Falls keine IFC-Datei angegeben: `USE_DEMO_MODEL = True`.

### 2. Notebook kopieren und anpassen

```
Vorlage: JaxFEM/Notebooks/IFC_SceneGraph_FEM.ipynb
Kopie:   JaxFEM/Notebooks/IFC_SG_FEM_<Name>_<Datum>.ipynb
```

Parameter in Schritt~1 (Zelle 4) setzen:
- `IFC_PATH`, `MESH_SIZE`, `Q_NUTZLAST`, `E_DEFAULT`, `NU_DEFAULT`
- `USE_DEMO_MODEL`

### 3. Notebook ausführen

```bash
cd /Users/philippschafer/Projekte/JaxFEM
conda activate jaxfem
jupyter nbconvert --to notebook --execute --inplace \
    Notebooks/IFC_SG_FEM_<Name>_<Datum>.ipynb \
    --ExecutePreprocessor.timeout=600
```

### 4. Ergebnisse auswerten und berichten

Claude liest die Ausgabe der Zusammenfassungs-Zelle (Schritt 20) und LLM-Prompt (Schritt 19) und gibt Philipp folgende Einschätzung:

#### a) Kenngrößen-Tabelle
| Größe | Wert | Einordnung |
|-------|------|-----------|
| u_z_max [mm] | — | Vergleich EC l/300 |
| σ_vM_max [kN/m²] | — | Vergleich f_ck (Beton) |
| σ_zz_min [kN/m²] | — | Druckspannung |
| Tet4-Elemente | — | Netzqualität |
| DOF | — | Systemgröße |

#### b) Normative Einordnung

- **Durchbiegung:** $u_{z,\text{max}} \leq L/300$ nach DIN EN 1990?
- **Druckspannung:** $|\sigma_{zz}| \leq f_{cd}$ nach EC2?
- **Von-Mises:** $\sigma_\text{vM} \leq f_y$ (bei Stahl)?

#### c) Handlungsempfehlung

Claude gibt an:
- Ob das Modell als **plausibel** einzustufen ist
- Welche **Bauteile kritisch** sind (höchste σ_vM)
- Ob **Netzverfeinerung** empfohlen wird
- Nächste Schritte

### 5. fem-engineer.js synchronisieren

Falls neue IFC-Logik oder Parameter geändert wurden → `fem-engineer.js` in  
`/Users/philippschafer/Projekte/CesiumJS/` entsprechend aktualisieren (analog `berechne_JAX`-Workflow).

---

## Ausgabedateien

| Datei | Pfad | Inhalt |
|-------|------|--------|
| Notebook | `JaxFEM/Notebooks/IFC_SG_FEM_<Name>_<Datum>.ipynb` | Ausgeführtes Notebook |
| JSON | `JaxFEM/data/ifc_scene_graph_fem.json` | Graph + FEM-Attribute |
| VTK | `JaxFEM/data/vtk/ifc_sg_fem.vtu` | ParaView/PyVista |

---

## Wichtige Parameter-Faustregeln

| Anwendungsfall | MESH_SIZE | Laufzeit |
|----------------|-----------|---------|
| Schnellcheck / Debug | 1,0 m | < 2 min |
| Vorentwurf | 0,5 m | 5–10 min |
| Belastbare Ergebnisse | 0,3 m | 15–30 min |
| Feine Auflösung | 0,1 m | > 1 h |

---

## Bekannte Fallstricke

1. **JIT-Kompilierung:** Erster Solver-Aufruf dauert 2–5 min — das ist kein Fehler.
2. **IFC-Einheiten:** Immer `ifcopenshell.util.unit.calculate_unit_scale` prüfen — IFC2x3 oft in mm statt m.
3. **Offene Netz-Geometrien:** Manche IFC-Bauteile liefern keine geschlossene STL → GMSH-Vernetzung schlägt fehl → Fallback auf BBox-Netz.
4. **IFC-Kanten:** `IfcRelAggregates` liefert nur Kanten, wenn Psets vollständig befüllt sind — bei vielen Demo-IFC-Dateien sind das 0 IFC-Kanten (rein geometrischer Graph).

---

## Referenz-Ergebnis (Meilenstein 1, April 2026)

Modell: `Ifc2x3_Duplex_Architecture.ifc`

```
IFC-Bauteile : 149
Graph-Knoten : 13
Graph-Kanten : 24  (IFC:0 / Geo:24)
Tet4-Elemente: 1958
DOF          : 1917
E_eff        : 30 000 000 kN/m²
Gesamtlast   : 21,81 kN/m²
u_z_max      : -2,064 mm
|u|_max      : 2,065 mm
σ_vM_max     : 7 529,7 kN/m²
σ_zz_min     : -9 072,6 kN/m²  (Druck)
```

---

## Verbindung zu anderen Skills

- `berechne_JAX` / `JAX it` → Templates für Einzelbauteile (Träger, Stütze, Scheibe, Platte)
- `IFC_FEM_3D.ipynb` → Vorgänger (BBox-Netz, manuelle BCs)
- `fem-engineer.js` → CesiumJS-Integration (geobim.app)
