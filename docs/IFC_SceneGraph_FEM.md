# IFC → Scene Graph → JaxFEM

**Meilenstein-Dokumentation — Version 1.0 | April 2026**

> **Status:** Erster vollständiger Durchlauf erfolgreich abgeschlossen.  
> IFC-Gebäudemodell → semantischer Scene Graph → automatische FEM-Randbedingungen → Tet4-Solver → Ergebnisrückschreibung in den Graph → LLM-Prompt.

---

## Überblick

`IFC_SceneGraph_FEM.ipynb` ist das bisher komplexeste Notebook in der JaxFEM-Pipeline. Es kombiniert vier bisher getrennte Welten:

| Welt | Technologie |
|------|-------------|
| BIM / IFC | IfcOpenShell |
| Räumliche KI | NetworkX DiGraph, DBSCAN (scikit-learn) |
| FEM-Vernetzung | GMSH Tet4 |
| Strukturmechanik | JaxFEM (JAX, lineare 3D-Elastizität) |

Die Kernidee: **Nicht nur Geometrie aus dem IFC extrahieren, sondern die semantischen Relationen des Gebäudes als Graph abbilden** — und diesen Graph direkt zur automatischen Ableitung von Randbedingungen und Lastflächen nutzen.

---

## Pipeline-Überblick

```
IFC-Datei (IfcOpenShell)
   └─ Bauteil-Parsing (IfcWall, IfcColumn, IfcSlab, IfcBeam, IfcFooting, IfcPile)
       ├─ BBox + Schwerpunkt (Weltkoordinaten, m)
       ├─ Material (E, ν aus Pset_MaterialMechanical oder Fallback)
       └─ IFC-Relationship-Analyse
           ├─ IfcRelAggregates          → contains / part_of
           ├─ IfcRelContainedInSpatialStructure → hosted_in
           ├─ IfcRelAssociatesMaterial  → shared_material
           └─ BBox-Spatial-Analyse      → above / adjacent / near / inside
               └─ Semantisches DBSCAN-Clustering (pro Label)
                   └─ Feature-Extraktion (Volumen, Höhe, Kompaktheit)
                       └─ NetworkX DiGraph (Scene Graph)
                           ├─ Graphanalyse → Randbedingungen + Lastflächen
                           ├─ Material-Mapping (volumengewichtetes E_eff, ν_eff)
                           └─ GMSH Tet4-Volumennetz (echte IFC-Geometrie)
                               └─ JaxFEM Solver (lineare 3D-Elastizität)
                                   └─ Ergebnisse → Scene Graph → LLM-Prompt
```

**Einheiten:** kN, m durchgängig.

---

## Konfigurationsparameter

Alle Parameter befinden sich in Schritt 1 und sind zentral steuerbar:

| Parameter | Standardwert | Bedeutung |
|-----------|-------------|-----------|
| `USE_DEMO_MODEL` | `False` | `True` = synthetisches Demo-IFC, `False` = eigene Datei |
| `IFC_PATH` | Duplex Architecture IFC 2x3 | Pfad zur eigenen IFC-Datei |
| `MESH_SIZE` | `1.0` m | Tet4-Elementgröße (kleiner = genauer, aber langsamer) |
| `E_DEFAULT` | `30 000 000` kN/m² | Fallback E-Modul (C25/30) wenn kein `Pset_MaterialMechanical` |
| `NU_DEFAULT` | `0.2` | Fallback Querdehnzahl |
| `GAMMA_BETON` | `25.0` kN/m³ | Wichte Stahlbeton |
| `Q_NUTZLAST` | `5.0` kN/m² | Charakteristische Nutzlast |
| `DBSCAN_EPS` | `0.6` m | Clustering-Radius für semantisches DBSCAN |
| `DBSCAN_MIN_SAMPLES` | `5` | Mindestpunkte pro DBSCAN-Cluster |
| `REL_THRESHOLD` | `5.0` m | Max. Distanz für Graphkante (geometrisch) |
| `ADJ_TOLERANCE` | `0.05` m | BBox-Berührungs-Toleranz für `adjacent`-Kante |

**Semantisches Mapping:**

```python
IFC_LABEL_MAP = {
    'IfcWall', 'IfcWallStandardCase' → 'wall'
    'IfcColumn'                      → 'column'
    'IfcSlab'                        → 'slab'
    'IfcBeam'                        → 'beam'
    'IfcFooting'                     → 'footing'
    'IfcPile'                        → 'pile'
}
```

---

## Schritt-für-Schritt-Beschreibung

### Schritt 2 — IFC laden

**Demo-Modus:** Erzeugt ein synthetisches IFC mit 2 Wänden + 1 Stütze + 1 Decke, inklusive `Pset_MaterialMechanical` (E, ν). Dient als reproduzierbarer Testfall ohne externe Dateiabhängigkeit.

**Produktiv-Modus:** Liest beliebige IFC 2x3/4-Dateien über IfcOpenShell. Getestet mit `Ifc2x3_Duplex_Architecture.ifc` (149 Bauteile).

### Schritt 3 — Bauteil-Parsing

Für jedes tragwerksrelevante Element:
1. Semantisches Label aus `IFC_LABEL_MAP`
2. Achsenparallele Bounding Box (BBox) aus Triangulierung
3. Material-Properties: `Pset_MaterialMechanical` → E-Modul, Querdehnzahl (Fallback: Betonkennwerte)

### Schritt 4 — Punktwolken-Visualisierung

Semantische Punktwolke (Schwerpunkte aller Bauteile) in XZ- und YZ-Schnittbildern, eingefärbt nach Bauteiltyp. Dient als schnelle Sanity-Check-Visualisierung.

### Schritt 5 — IFC-Relationship-Analyse

Drei Kantenquellen:

| IFC-Relationship | Kanten-Typ | Bedeutung |
|---|---|---|
| `IfcRelAggregates` | `part_of` / `contains` | Hierarchische Zerlegung |
| `IfcRelContainedInSpatialStructure` | `hosted_in` | Stockwerk-Zuordnung |
| `IfcRelAssociatesMaterial` | `shared_material` | Gleicher Werkstoff |
| BBox-Spatial (Schritt 8) | `above/adjacent/near` | Geometrische Lage |

### Schritt 6 — Semantisches Clustering (DBSCAN)

Analog Poux & Lehtola (2025): DBSCAN-Clustering auf Schwerpunkten pro semantischem Label. Mehrere räumlich getrennte Instanzen eines Bauteiltyps (z.B. zwei Wandgruppen) werden als separate Graph-Knoten repräsentiert.

Parameter: `eps = DBSCAN_EPS`, `min_samples = DBSCAN_MIN_SAMPLES`. Rauschpunkte (Cluster-ID = -1) werden direkt als Einzelinstanzen übernommen.

### Schritt 7 — Feature-Extraktion

Pro Objekt-Instanz (nach Clustering):

| Feature | Berechnung |
|---------|-----------|
| `volume` | BBox-Volumen (m³) |
| `surface_area` | Konvexe Hülle (scipy.spatial.ConvexHull), m² |
| `compactness` | V^(2/3) / A — dimensionsloser Formfaktor |
| `height` | z_max − z_min (m) |
| `aspect_ratio` | max(dx,dy) / dz |

### Schritt 8 — Räumliche Beziehungsanalyse

Drei-Stufen-Algorithmus auf BBox-Basis:

1. **Vertikal:** `above`/`below` wenn Höhenunterschied > 0,5 m
2. **Containment:** `inside` / `contains` per BBox-Einschachtelung
3. **Adjazenz:** `adjacent` wenn BBox-Abstand < `ADJ_TOLERANCE`; `near` wenn < `REL_THRESHOLD`

### Schritt 9 — Scene Graph (NetworkX DiGraph)

Jeder **Knoten** trägt:
- Semantisches Label, IFC-Name, GUID
- BBox (bounds_min, bounds_max), Schwerpunkt, Volumen, Höhe
- E-Modul [kN/m²], Querdehnzahl ν
- Feature-Werte (Kompaktheit, Aspektverhältnis)

Jede **Kante** trägt:
- `relationship` (Typ, s.o.)
- `source` = `'ifc'` oder `'geometric'`

### Schritt 11 — Graphanalyse: Randbedingungen + Lastflächen

| Graph-Eigenschaft | FEM-Ableitung |
|---|---|
| `wall`/`column` ohne eingehende `below`-Kante | Einspannung am Fuß (z = z_min) |
| `slab` mit ausgehender `above`-Kante zu Wand/Stütze | Lastfläche |
| `slab` auf höchster z-Ebene | Primäre Lastauflagefläche |

### Schritt 12 — Material-Mapping

Volumengewichtetes effektives Material über alle Graphknoten:

```
E_eff  = Σ(E_i · V_i) / Σ V_i
ν_eff  = Σ(ν_i · V_i) / Σ V_i
```

Daraus: Lamé-Parameter λ und μ für den FEM-Solver.

### Schritt 13 — GMSH Tet4-Volumnetz

Strategie: **Echte IFC-Dreiecksgeometrie** (via `ifcopenshell.geom`) als GMSH-STL-Eingang → Discrete-Surface-Modell → 3D-Vernetzung mit Tet4-Elementen.

```
mesh_size = MESH_SIZE  [m]
out_msh   = '../data/msh/ifc_sg.msh'
```

Randbedingungs-Marker werden direkt beim Netzaufbau als physikalische Gruppen gesetzt.

### Schritt 14 — Randbedingungen aus Bauteilkontakt

Material-basierte Kontakt-BCs aus dem Scene Graph:

| Verbindung | BC-Typ | Begründung |
|---|---|---|
| Stahlbeton ↔ Stahlbeton | Eingespannt (u_x=u_y=u_z=0) | Monolithischer Verbund |
| mind. 1 Holzbauteil | Gelenkig (u_x=u_y=u_z=0, keine Momente) | Holz-Verbindungen → kein Momentenübertrag |
| Fundament-Ebene (z=z_min) | Immer eingespannt | Bodenhaftung |
| Lastfläche (z=z_top) | Flächenlast q | Eigengewicht + Nutzlast |

### Schritt 15 — JaxFEM-Solver

Lineare 3D-Elastizität (Hooke'sches Gesetz, isotropes Material):

```python
class IFC_SG_FEM(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
        return stress

    def get_surface_maps(self):
        def load_top(u, x):
            return jnp.array([0.0, 0.0, q])  # Flächenlast in z
        return [load_top]
```

Löser: `umfpack_solver` (direkt, empfohlen für <10 000 DOF).

### Schritt 16 — Spannungsberechnung (Tet4)

Manuelle Tet4-Spannungsberechnung (unabhängig vom JaxFEM-Solver-Output):

1. Jacobi-Matrix J aus Knotenkoordinaten
2. Formfunktions-Ableitungen `dNdx = dN @ J_inv`
3. Verzerrungstensor: `eps = 0.5*(grad_u + grad_u.T)`
4. Spannungstensor: `sigma = lmbda*tr(eps)*I + 2*mu*eps`
5. Von-Mises-Spannung: `σ_vM = sqrt(1.5 * dev(σ):dev(σ))`

### Schritt 18 — Ergebnisse in Scene Graph zurückschreiben

Jeder Graphknoten erhält FEM-Ergebnisse aus dem zugehörigen BBox-Bereich:

| Attribut | Bedeutung |
|----------|-----------|
| `uz_min_mm` | Minimale z-Verschiebung im Bauteil (mm) |
| `u_abs_max_mm` | Maximale Gesamtverschiebung (mm) |
| `sigma_vm_max` | Maximale Von-Mises-Spannung (kN/m²) |

### Schritt 19 — LLM-Prompt

Der angereicherte Scene Graph (IFC-Relationen + geometrische Relationen + FEM-Ergebnisse) wird in einen strukturierten Textprompt serialisiert:

```
=== IFC TRAGWERK — SCENE GRAPH + FEM-ERGEBNISSE ===

BAUTEILE (Knoten):
  [wall_0]  IFC=BasicWall:... (IfcWall)  Typ=wall  
            Schwerpunkt=(x, y, z) m  Volumen=... m³
            E=30000000 kN/m²  uz_min=-1.2 mm  σ_vM_max=4200 kN/m²

BEZIEHUNGEN (Kanten):
  wall_0 → slab_0  [adjacent | geometric]
  ...
```

Dieser Prompt kann direkt an Claude oder ein anderes LLM übergeben werden, um strukturmechanische Fragen automatisch zu beantworten.

---

## Ergebnisse — Erster Meilenstein (April 2026)

Getestet mit: `Ifc2x3_Duplex_Architecture.ifc`

| Kenngröße | Wert |
|-----------|------|
| IFC-Bauteile (gesamt) | 149 |
| Graph-Knoten (Instanzen) | 13 |
| Graph-Kanten | 24 (IFC: 0 / Geo: 24) |
| Tet4-Elemente | 1 958 |
| FEM-Knoten | 639 |
| Freiheitsgrade (DOF) | 1 917 |
| E_eff | 30 000 000 kN/m² |
| ν_eff | 0,20 |
| Gesamtlast | 21,81 kN/m² |
| **u_z_max** | **−2,064 mm** |
| **\|u\|_max** | **2,065 mm** |
| **σ_vM_max** | **7 529,7 kN/m²** |
| **σ_zz_min (Druck)** | **−9 072,6 kN/m²** |

Ausgabedateien:
- `../data/ifc_scene_graph_fem.json` — vollständiger Graph mit FEM-Attributen
- `../data/vtk/ifc_sg_fem.vtu` — VTK-Ergebnisdatei (ParaView / PyVista)

---

## Was ist neu gegenüber `IFC_FEM_3D.ipynb`

| Merkmal | IFC_FEM_3D.ipynb | IFC_SceneGraph_FEM.ipynb |
|---------|-----------------|--------------------------|
| Netztyp | GMSH Tet4 (BBox-basiert) | GMSH Tet4 (echte IFC-Triangulierung) |
| Randbedingungen | Manuell / BBox-Ebenen | **Automatisch aus Scene Graph** |
| Lastflächen | Manuell | **Automatisch aus Graph-Topologie** |
| Material | Manuell | **Automatisch aus IFC + Fallback** |
| Struktursemantik | Keine | **DiGraph mit 9 Kanten-Typen** |
| LLM-Integration | Keine | **Strukturierter Prompt-Generator** |
| Ergebnisrückfluss | Keine | **FEM → Graphknoten** |

---

## Bekannte Einschränkungen (Version 1.0)

- IFC-Relationen (`IfcRelAggregates`) werden im Duplex-Beispiel nicht ausgelesen (→ 0 IFC-Kanten); alle 24 Kanten sind geometrisch. Dies liegt am spezifischen IFC-Modell, nicht am Algorithmus.
- `MESH_SIZE = 1.0 m` ist grob (Debugging-Einstellung) — für Produktionsberechnungen `0.2–0.5 m` verwenden.
- Volumengewichtetes E_eff / ν_eff: vereinfachte Homogenisierung. Für Verbundsysteme ist eine bauteilweise Materialdefinition anzustreben.
- LLM-Prompt-Ausgabe bisher nur für OpenAI/Anthropic API vorbereitet; kein Auto-Call implementiert.

---

## Abhängigkeiten

```
ifcopenshell >= 0.8
jax, jaxlib
numpy
networkx
scikit-learn       (DBSCAN)
scipy              (ConvexHull)
gmsh
meshio
pyvista
matplotlib
```

Conda-Umgebung: `jaxfem` (siehe `environment.yml` im JaxFEM-Root).

---

## Nächste Entwicklungsschritte

| Priorität | Schritt |
|-----------|---------|
| Hoch | `MESH_SIZE` auf 0,3 m reduzieren, Konvergenztest |
| Hoch | Bauteilweises Material im FEM-Solver (heterogenes E-Feld) |
| Hoch | IFC-Relationen aus realen IFC4-Dateien mit befüllten Psets aktivieren |
| Mittel | LLM-Auto-Call: Graph-Prompt → Claude API → Strukturbewertung |
| Mittel | Ergebnis-Export als GeoJSON → CesiumJS-Einfärbung |
| Forschung | Gradient-basierte Formoptimierung via JAX (differenzierbarer Solver) |

---

## Referenzen

- Poux, F. & Lehtola, V. (2025). *3D Scene Graphs for Spatial AI and LLMs*. Data Science Collective.
- IFC_FEM_3D.ipynb — BBox-Netz + JaxFEM 3D (Vorgänger-Notebook)
- SceneGraph_FEM.ipynb — Synthetischer Scene-Graph-Workflow
- JaxFEM: https://github.com/tianjuxue/jax-fem
- IfcOpenShell: https://ifcopenshell.org
