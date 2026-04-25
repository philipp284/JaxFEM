---
title: "JaxFEM Notebooks — BIM-native FEM-Templates für das Bauingenieurwesen"
type: project
scope: research + development + teaching
status: active
last_updated: "2026-04-25"
technology: [Python, JaxFEM, JAX, IfcOpenShell, GMSH, meshio, PyVista, IFC4, HEX8, TET4, Eurocode, NetworkX, DBSCAN, SceneGraph, LLM-Prompt]
domain: [BIM, Strukturmechanik, FEM, Bauingenieurwesen, Lehre, Digitales Bauen]
institution: Hochschule Mainz — Tragwerkslabor
related:
  - path: /Users/philippschafer/Projekte/JaxFEM/CLAUDE.md
    label: JaxFEM Projekt-Root
  - path: /Users/philippschafer/Projekte/CesiumJS/CLAUDE.md
    label: CesiumJS / geobim.app
  - path: /Users/philippschafer/Library/CloudStorage/OneDrive-HochschuleMainz-UniversityofAppliedSciences/03_Tragwerkslabor/06_Projekte/2026_03_11_MIB/CLAUDE.md
    label: MIB-Forschungsprojekt
---

# JaxFEM Notebooks — BIM-native FEM-Templates für das Bauingenieurwesen

## Überblick

Dieses Verzeichnis enthält zwei Kategorien von Notebooks:

1. **Pipeline-Notebooks** — IFC → Mesh → JaxFEM-Solver (CesiumJS/geobim.app-Integration)
2. **Lehr-/Forschungstemplates** — vollständige FEM-Berechnungen typischer Bauteile mit Eurocode-Materialien

---

## Lehr- und Forschungstemplates (Bauingenieurwesen)

### Werkstoffkennwerte nach Eurocode (für alle Templates)

Alle Templates verwenden das Modul `ec_materials.py` im `Notebooks/`-Ordner. Die Kennwerte sind in `kN + m` normiert (JaxFEM-Standard).

| Werkstoff | Norm | E-Modul [kN/m²] | Poissonzahl ν | Rohdichte ρ [kN/m³] |
|-----------|------|-----------------|---------------|----------------------|
| Stahl S235 | EN 1993 | 210 000 000 | 0,30 | 78,5 |
| Stahl S355 | EN 1993 | 210 000 000 | 0,30 | 78,5 |
| Beton C20/25 | EN 1992 | 30 000 000 | 0,20 | 25,0 |
| Beton C25/30 | EN 1992 | 31 000 000 | 0,20 | 25,0 |
| Beton C30/37 | EN 1992 | 32 000 000 | 0,20 | 25,0 |
| Beton C40/50 | EN 1992 | 35 000 000 | 0,20 | 25,0 |
| Holz GL24h | EN 1995 | 11 600 000 | 0,40 | 4,2 |
| Holz GL32h | EN 1995 | 13 700 000 | 0,40 | 4,4 |
| Mauerwerk MZ (M10) | EN 1996 | 5 000 000 | 0,20 | 18,0 |
| Mauerwerk KS (M10) | EN 1996 | 8 000 000 | 0,20 | 19,0 |

### Templates — Übersicht

| Notebook | Bauteil | Materialien | Besonderheit |
|----------|---------|-------------|--------------|
| `Template_Traeger.ipynb` | Einfeldträger (Deckenbalken) | Stahl S235/S355, Beton C25/30, Holz GL24h, Verbund, Mauerwerk | IFC → box_mesh → HEX8, PyVista 3D |
| `Template_Stuetze.ipynb` | Rechteckstütze (Drucklast + Imperfektion) | Stahl S235, Beton C25/30, Holz GL24h | IFC → box_mesh → HEX8, PyVista 3D |
| `Template_Scheibe.ipynb` | Wandscheibe (ebener Spannungszustand) | Stahl S235, Beton C25/30, Mauerwerk | IFC → box_mesh → HEX8, PyVista 3D |
| `Template_Platte.ipynb` | Zweiachsig gespannte Platte | Beton C25/30 | IFC → box_mesh → HEX8, PyVista 3D |
| `Template_Plattenbalken.ipynb` | Plattenbalken mit Stahlbetonbewehrung | Beton C25/30 + Betonstahl B500B | IFC T-Querschnitt → HEX8, Bewehrungslage, PyVista 3D |

### Workflow-Konvention (alle Templates)

```
Schritt 1 — IFC-Modell erzeugen (IfcOpenShell)
  └─ _box_vf() → Quader-Geometrie
  └─ IfcBeam / IfcColumn / IfcSlab / IfcWall

Schritt 2 — IFC → JaxFEM-Netz
  └─ bbox pro Bauteil → box_mesh() → HEX8
  └─ Bauteilweise Verschiebung (x0, y0, z0)

Schritt 3 — Material (Eurocode, kN + m)
  └─ ec_materials.py laden oder inline definieren

Schritt 4 — Problem-Klasse (LinearElasticity3D)
  └─ get_tensor_map(): Hooke'sches Gesetz
  └─ get_surface_maps(): Flächenlasten

Schritt 5 — Randbedingungen
  └─ Dirichlet: Einspannung / Gelenk
  └─ Neumann: Flächenlast / Linienlast

Schritt 6 — Solver (umfpack_solver)

Schritt 7 — Ergebnisse
  └─ Matplotlib: Draufsicht, Schnitt, Schnittgrößen
  └─ PyVista: 3D-Visualisierung, Spannungsfärbung

Schritt 8 — Analytischer Vergleich (EC-Formeln)
```

### Wichtige Hinweise für Template-Erstellung

- **Einheiten:** kN und m durchgängig (E-Modul in kN/m² = MPa × 1000)
- **IFC-Einheiten:** `ifcopenshell.api.run('unit.assign_unit', model, length={'is_metric': True, 'raw': 'METRES'})`
- **PyVista:** `pv.set_jupyter_backend('static')` für PNG im Notebook
- **Spannungsextraktion:** `compute_stress_at_nodes_3d()` aus Flachdecke.ipynb als Vorlage
- **Analytischer Vergleich:** immer mit Euler-Bernoulli / Kirchhoff / Eurocode-Tabellenwerten

---

# CesiumJS → IFC → JaxFEM Pipeline

## Vision

**Ein Ingenieur öffnet geobim.app, wählt ein Bauteil oder ein Gebäude in der 3D-Ansicht an — und bekommt automatisch eine strukturmechanische FEM-Berechnung zurück.**

Das ist kein Konzept. Das wird hier gebaut.

Die Pipeline verbindet drei bisher getrennte Welten:

```
CesiumJS (geobim.app)          IFC-Geometrie              JaxFEM
  BIM-Viewer in Geo-           ─────────────────►    JAX-basierter FEM-Solver
  Kontext (Cesium Ion,          IfcOpenShell +         GPU-beschleunigt,
  3D Tiles, Firebase)           GMSH + meshio          differenzierbar
```

---

## Was hier gebaut wird

### Gesamtarchitektur

```
geobim.app (Browser)
  └─ IFC-Datei liegt in Firebase Storage / Hetzner-Server
       └─ [Download per URL oder direkter Pfad]
            └─ IfcOpenShell (Python)
                 ├─ IFC parsen: IfcBeam, IfcColumn, IfcSlab, IfcWall
                 ├─ Geometrie extrahieren (Vertices + Faces, Weltkoordinaten)
                 └─ Material-Properties (IfcMaterial → E, nu, rho)
                      └─ GMSH (wahlweise) oder box_mesh (JaxFEM)
                           ├─ Tet4-Volumen-Netz  (GMSH, für echte Geometrien)
                           └─ HEX8-Box-Netz      (Fallback, aus BBox, schnell)
                                └─ JaxFEM
                                     ├─ Problem-Klasse (lineare Elastizität 3D)
                                     ├─ Randbedingungen (Einspannung, Auflager)
                                     ├─ Solver (umfpack / JAX-BiCGSTAB)
                                     └─ Ergebnisse:
                                          ├─ Verschiebungen (u_x, u_y, u_z) [mm]
                                          ├─ Spannungen (σ_xx, σ_vM) [kN/m²]
                                          └─ Export:
                                               ├─ VTK (.vtu) → ParaView
                                               └─ JSON → geobim.app Einfärbung
                                                         (σ_vM pro Bauteil-GUID)
```

---

## Notebooks in diesem Ordner

| Notebook | Inhalt | Status |
|----------|--------|--------|
| `IFC_Mesh_Demo.ipynb` | **Startpunkt.** IFC-Modell aufbauen (2 Wände + Stütze + Decke 4×4m), HEX8-Netz erzeugen, 3D-Visualisierung mit Matplotlib + PyVista | ✅ läuft vollständig |
| `GESAMTSTATIK.ipynb` | Vollständige Pipeline: IFC → GMSH/Fallback → JaxFEM → Ergebnisse + JSON-Export | ⚠️ Pipeline OK, Solver braucht Tuning |
| `Rechteckbalken_3D.ipynb` | Validierung: Einfeldträger HEX8, Vergleich mit Euler-Bernoulli | ✅ |
| `Traeger_Statik.ipynb` | 2D-Scheibe: EFK / ZF / DSL, Schnittgrößen M/V | ✅ |
| `Flachdecke.ipynb` | Flachdecke 2-Way-Slab | ✅ |
| `Stuetze_6m.ipynb` | Druckstab / Knicklast | ✅ |

---

## Technische Details

### Einheiten
- **Kräfte:** kN  
- **Längen:** m  
- **E-Modul:** kN/m²  (Beton C25/30 = 31 000 kN/m²)
- **Spannungen:** kN/m²

### IFC-Einheiten-Falle
IFC-Standard speichert Geometrie in **mm** (nicht m).  
`IFC_UNIT_TO_M = 0.001` ist der Umrechnungsfaktor — bei IFC4-Modellen, die explizit METRES setzen, ist er `1.0`.  
Immer prüfen via `ifcopenshell.util.unit.calculate_unit_scale(model, 'LENGTHUNIT')`.

### IfcOpenShell API-Versionsnotiz (0.8.x)
```python
# products ist eine Liste — nicht product=elem
ifcopenshell.api.run('aggregate.assign_object', model,
                      products=[elem], relating_object=parent)

# add_box_representation existiert nicht → add_mesh_representation verwenden
v, f = _box_verts_faces(x0, y0, z0, dx, dy, dz)
ifcopenshell.api.run('geometry.add_mesh_representation', model,
                      context=body, vertices=[v], faces=[f])
```

### JaxFEM Mesh-Keys
```python
m = box_mesh(Nx=nx, Ny=ny, Nz=nz, domain_x=Lx, domain_y=Ly, domain_z=Lz)
# Korrekte Schlüssel (meshio-Standard):
cells = m.cells_dict['hexahedron']   # HEX8
# NICHT: 'hexahedron8' — das wirft KeyError
```

### GMSH-Strategie
- **Weg A (empfohlen für echte IFC):** IfcOpenShell exportiert STEP/BRep → GMSH importiert über OCC-Kernel → geschlossenes Tet4-Volumen
- **Weg B (Demo/Fallback):** IFC-Triangulierung als Discrete Surface → GMSH classifySurfaces → oft nicht dicht geschlossen
- **Weg C (aktuell implementiert):** `box_mesh` aus BBox jedes Bauteils → immer robust, geringere Genauigkeit an Kontaktzonen

### Solver-Konvergenz
- JaxFEM JIT-Kompilierung beim ersten Aufruf: **2–5 Minuten** — das ist kein Fehler, kein Endless-Loop
- Bei großen Modellen (>5000 DOF): `umfpack_solver` bevorzugen
- Separates Testen pro Bauteil (z.B. nur Deckenplatte) vor Gesamtmodell

---

## Nächste Entwicklungsschritte

| # | Schritt | Priorität |
|---|---------|-----------|
| 1 | **Solver-Test:** Nur Deckenplatte, Einfeldplatte gelenkig gelagert, Vergleich Analytik | hoch |
| 2 | **GMSH-Tet4 via STEP:** IfcOpenShell → `shape.geometry` → OCC → GMSH | hoch |
| 3 | **IFC-Einheiten automatisch:** `ifcopenshell.util.unit` statt Hardcode | mittel |
| 4 | **Material aus IFC:** `IfcMaterialLayerSet` → E-Modul-Mapping | mittel |
| 5 | **Randbedingungen aus IFC:** `IfcStructuralSupportCondition` | mittel |
| 6 | **Server-Skript** `server/ifc_to_fem.py`: REST-Endpoint für geobim.app | hoch |
| 7 | **CesiumJS-Integration:** JSON-Ergebnis → Property-Table → Bauteil-Einfärbung nach σ_vM | hoch |
| 8 | **Topologie-Optimierung** auf importierter Geometrie | Forschung |

---

## Warum das ein Gamechanger ist

Heute braucht ein Tragwerksplaner für eine FEM-Berechnung:
1. BIM-Modell aus Revit/Archicad exportieren
2. Manuelle Import-Aufbereitung in RFEM / ANSYS (Stunden bis Tage)
3. Randbedingungen von Hand setzen
4. Netz von Hand definieren
5. Ergebnisse manuell zurück ins BIM-Modell übertragen

**Mit dieser Pipeline:**
1. IFC aus geobim.app auswählen (ein Klick)
2. Pipeline läuft automatisch durch
3. Ergebnisse erscheinen als Farbskala direkt im Geo-BIM-Viewer

Das ist **BIM-native strukturmechanische Analyse** — vollständig in Open-Source, vollständig automatisiert, direkt im Browser.

Langfristig: JAX macht den Solver **differenzierbar** — damit wird aus der reinen Analyse eine **automatische Formoptimierung** möglich: Welche Wandstärke minimiert Verformungen bei gegebenem Materialvolumen? JAX beantwortet das mit einem einzigen Gradientenabstieg.

---

## Demo-Modell (IFC_Mesh_Demo.ipynb)

```
Gebäude (EG):
  ├── Wand_A    (IfcWall)    x=0–4m, y=0–0.2m,   z=0–2.8m
  ├── Wand_B    (IfcWall)    x=0–4m, y=3.8–4.0m,  z=0–2.8m
  ├── Stütze_M  (IfcColumn)  x=1.9–2.1m, y=1.9–2.1m, z=0–2.8m
  └── Decke_D1  (IfcSlab)    x=0–4m, y=0–4m,      z=2.8–3.0m

Netz (h=0.2m):
  974 HEX8-Elemente  |  2202 Knoten  |  6606 DOF

Dateien:
  data/demo_statik.ifc              — IFC4-Quelldatei
  data/vtk/demo_mesh_3d.png         — 3D-Netz-Plot
```
