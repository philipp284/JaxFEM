# JAX-FEM – Vollständige Einsteiger-Dokumentation

Dieses Dokument erklärt, was JAX-FEM ist, wie es funktioniert und wie du es Schritt für Schritt benutzt – sowohl als einfaches Python-Skript als auch interaktiv in Jupyter Notebook. Vorkenntnisse in der Finite-Elemente-Methode sind hilfreich, aber nicht zwingend erforderlich.

---

## Inhaltsverzeichnis

1. [Was ist JAX-FEM?](#1-was-ist-jax-fem)
2. [Was ist die Finite-Elemente-Methode?](#2-was-ist-die-finite-elemente-methode)
3. [Voraussetzungen und Umgebung](#3-voraussetzungen-und-umgebung)
4. [Projektstruktur im Überblick](#4-projektstruktur-im-überblick)
5. [Die Bausteine eines JAX-FEM-Programms](#5-die-bausteine-eines-jax-fem-programms)
6. [Methode A: Python-Skript direkt ausführen](#6-methode-a-python-skript-direkt-ausführen)
7. [Methode B: Interaktiv mit Jupyter Notebook](#7-methode-b-interaktiv-mit-jupyter-notebook)
8. [Ergebnisse visualisieren](#8-ergebnisse-visualisieren)
9. [Unterstützte Elementtypen](#9-unterstützte-elementtypen)
10. [Randbedingungen verstehen](#10-randbedingungen-verstehen)
11. [Häufige Fehler und Lösungen](#11-häufige-fehler-und-lösungen)
12. [Ergebnis & Visualisierung – Spannungen, Dehnungen, Trajektorien](#12-ergebnis--visualisierung--spannungen-dehnungen-trajektorien)

---

## 1. Was ist JAX-FEM?

JAX-FEM ist eine Python-Bibliothek zur numerischen Simulation physikalischer Probleme – zum Beispiel Wärmeleitung, Strukturmechanik oder Strömungen. Sie basiert auf **JAX**, einem wissenschaftlichen Rechenpaket von Google, das automatische Differentiation und GPU-Beschleunigung bietet.

Der besondere Vorteil gegenüber klassischen FEM-Programmen (z. B. ANSYS, ABAQUS): Jede Simulation ist automatisch **differenzierbar**. Das bedeutet, man kann mit denselben Werkzeugen nicht nur ein Problem lösen, sondern auch automatisch optimieren – z. B. die beste Materialverteilung für eine Struktur finden (Topologieoptimierung).

**Typische Anwendungsfälle:**

- Wärmeleitung und Thermomechanik
- Lineare und nichtlineare Elastizität
- Phasenfeldmodelle (Risse, Erstarrung)
- Kristallplastizität
- Strömungsmechanik (Stokes-Gleichungen)
- Inverse Probleme und Design-Optimierung

---

## 2. Was ist die Finite-Elemente-Methode?

Die **Finite-Elemente-Methode (FEM)** ist ein numerisches Verfahren, um partielle Differentialgleichungen (PDEs) näherungsweise zu lösen. Statt eine exakte analytische Lösung zu suchen, wird das Rechengebiet in kleine Teilbereiche – die **finiten Elemente** – zerlegt. Auf jedem Element wird die Lösung durch einfache Ansatzfunktionen (z. B. linear oder quadratisch) approximiert.

### Der prinzipielle Ablauf:

```
1. Gebiet definieren (Geometrie)
        ↓
2. Vernetzung (Mesh) – Gebiet in Elemente einteilen
        ↓
3. Physik definieren – Welche PDE wird gelöst?
        ↓
4. Randbedingungen – Was ist am Rand fest vorgegeben?
        ↓
5. Gleichungssystem aufstellen und lösen
        ↓
6. Ergebnis auswerten und visualisieren
```

In JAX-FEM entspricht jeder dieser Schritte einem konkreten Python-Code-Abschnitt, der weiter unten erklärt wird.

---

## 3. Voraussetzungen und Umgebung

### Python-Umgebung aktivieren

Das Projekt enthält eine fertige virtuelle Umgebung im Ordner `.venv`. Alle notwendigen Pakete (JAX, Gmsh, Basix, meshio, Jupyter, ...) sind dort bereits installiert.

Du musst **nicht** selbst etwas installieren. Verwende immer den Python-Interpreter aus dieser Umgebung:

```bash
# Pfad zum Python-Interpreter der virtuellen Umgebung:
.venv/bin/python

# Pfad zu Jupyter:
.venv/bin/jupyter
```

> **Tipp:** Wenn du im Terminal bist, musst du dich immer im Projektordner `/Users/philippschafer/Projekte/JaxFEM` befinden. Du kannst dorthin wechseln mit:
> ```bash
> cd /Users/philippschafer/Projekte/JaxFEM
> ```

---

## 4. Projektstruktur im Überblick

```
JaxFEM/
│
├── jax_fem/                  ← Kernbibliothek (hier ist der eigentliche Code)
│   ├── problem.py            ← Basisklasse für alle FEM-Probleme
│   ├── solver.py             ← Newton-Solver und lineare Gleichungslöser
│   ├── generate_mesh.py      ← Netzgenerierung (Rechteck, Quader, Gmsh)
│   ├── fe.py                 ← Finite-Element-Berechnungen (Ansatzfunktionen, Quadratur)
│   ├── basis.py              ← Elementdefinitionen (QUAD4, HEX8, TET4, ...)
│   └── utils.py              ← Hilfsfunktionen (Speichern, Timing, ...)
│
├── applications/             ← Fertige Beispielanwendungen
│   ├── thermal_mechanical/   ← Thermomechanik
│   ├── topology_opt/         ← Topologieoptimierung
│   ├── stokes/               ← Strömungssimulation
│   └── ...                   ← viele weitere Beispiele
│
├── example.py                ← Einfaches 2D-Beispiel (Poisson-Gleichung)
├── simple_workflow.py        ← 3D-Beispiel mit Gmsh-Netz und PyVista-Visualisierung
├── einstieg.ipynb            ← Jupyter Notebook für interaktive Nutzung
│
├── data/                     ← Ausgabeordner für Ergebnisse
│   └── vtk/                  ← VTK-Dateien für ParaView / PyVista
│
└── .venv/                    ← Virtuelle Python-Umgebung (nicht bearbeiten)
```

---

## 5. Die Bausteine eines JAX-FEM-Programms

Ein vollständiges JAX-FEM-Programm besteht immer aus denselben fünf Bausteinen. Hier werden sie am Beispiel der **Poisson-Gleichung** erklärt – einer der einfachsten PDEs, die beschreibt, wie sich z. B. Wärme in einem Körper verteilt.

### Baustein 1: Netz erstellen (`generate_mesh`)

Das Netz (engl. *mesh*) legt fest, wie das geometrische Gebiet in Elemente unterteilt wird.

```python
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

ele_type = 'QUAD4'                             # Elementtyp: 4-Knoten-Viereck (2D)
cell_type = get_meshio_cell_type(ele_type)     # Internes Format für meshio

Lx, Ly = 1., 1.                               # Gebietsgröße: 1m × 1m
meshio_mesh = rectangle_mesh(Nx=32, Ny=32,    # 32×32 Elemente
                             domain_x=Lx, domain_y=Ly)

mesh = Mesh(meshio_mesh.points,               # Knotenkoordinaten
            meshio_mesh.cells_dict[cell_type]) # Elementverbindungen
```

**Was passiert hier?**
- `rectangle_mesh(Nx=32, Ny=32, ...)` erzeugt ein strukturiertes Gitter aus 32×32 = 1024 Viereck-Elementen.
- `meshio_mesh.points` enthält die x/y-Koordinaten aller Knoten als Array der Form `(Anzahl_Knoten, 2)`.
- `meshio_mesh.cells_dict[cell_type]` enthält die Knotenindizes jedes Elements als Array der Form `(Anzahl_Elemente, 4)`.
- Das `Mesh`-Objekt ist ein einfacher Container, der beide Arrays bündelt.

Für 3D-Probleme gibt es `box_mesh_gmsh(...)`, das über die externe Bibliothek **Gmsh** ein Hexaeder-Netz erzeugt:

```python
from jax_fem.generate_mesh import box_mesh_gmsh

meshio_mesh = box_mesh_gmsh(Nx=10, Ny=5, Nz=5,
                            domain_x=2.0, domain_y=1.0, domain_z=1.0,
                            data_dir='./data', ele_type='HEX8')
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
```

---

### Baustein 2: Physik definieren (`Problem`)

Die Klasse `Problem` aus `jax_fem/problem.py` ist das Herzstück. Hier wird die **schwache Form** der PDE definiert – das mathematische Rezept, nach dem JAX-FEM das Gleichungssystem aufbaut.

Man erstellt immer eine eigene Unterklasse von `Problem` und überschreibt bestimmte Methoden:

```python
import jax.numpy as np
from jax_fem.problem import Problem

class Poisson(Problem):

    def get_tensor_map(self):
        # Definiert den Diffusionsterm: ∇u (Gradient der Lösung)
        # Für einfache Laplace-/Poisson-Gleichung: Identitätsabbildung
        return lambda x: x

    def get_mass_map(self):
        # Definiert den Quellterm f(u, x) auf der rechten Seite der PDE
        def mass_map(u, x):
            # Gaußsche Wärmequelle in der Mitte des Gebiets
            val = -np.array([10 * np.exp(
                -(np.power(x[0] - 0.5, 2) + np.power(x[1] - 0.5, 2)) / 0.02
            )])
            return val
        return mass_map
```

**Was bedeuten diese Methoden?**

| Methode | Bedeutung | Wann überschreiben? |
|---|---|---|
| `get_tensor_map()` | Materialgesetz / Diffusionsterm | Immer – definiert die Physik |
| `get_mass_map()` | Quellterm / Körperkraft | Wenn externe Lasten vorhanden sind |
| `get_surface_maps()` | Neumann-Randbedingungen (Fluss) | Wenn Wärmefluss / Kraft am Rand |
| `set_params(params)` | Parameter für Optimierung setzen | Bei Inverse-Problemen |

Die Methode `get_tensor_map()` gibt eine Funktion zurück, die den **Gradienten der Lösung** auf den **Fluss** abbildet. Für die einfache Wärmeleitungsgleichung ist das:

```
Fluss = λ · ∇u   (Fouriersches Gesetz)
```

Mit `lambda x: x` (Identität) bedeutet das λ = 1 (normierte Wärmeleitfähigkeit).

---

### Baustein 3: Randbedingungen definieren

Randbedingungen legen fest, was an den Rändern des Gebiets gilt. Die häufigste Variante sind **Dirichlet-Randbedingungen**: Die Lösung hat an bestimmten Stellen einen fest vorgegebenen Wert.

```python
# Schritt 1: Ortsfunktionen – wo gilt die Randbedingung?
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)   # Linker Rand: x ≈ 0

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)   # Rechter Rand: x ≈ Lx

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)   # Unterer Rand: y ≈ 0

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)   # Oberer Rand: y ≈ Ly

# Schritt 2: Wertfunktion – welchen Wert hat die Lösung dort?
def dirichlet_val(point):
    return 0.   # u = 0 an allen Rändern

# Schritt 3: Alles zusammensetzen
location_fns = [left, right, bottom, top]  # An welchen Rändern?
vecs         = [0, 0, 0, 0]               # Welche Komponente? (0 = erste/einzige)
value_fns    = [dirichlet_val] * 4        # Welcher Wert?

dirichlet_bc_info = [location_fns, vecs, value_fns]
```

**Wichtig:** Jede `location_fn` bekommt einen Punkt `point` (Array mit x, y oder x, y, z Koordinaten) und gibt `True` zurück, wenn der Punkt auf dem entsprechenden Rand liegt.

---

### Baustein 4: Problem instanziieren und lösen

```python
from jax_fem.solver import solver

# Problem-Objekt erzeugen
problem = Poisson(
    mesh=mesh,                             # Das Netz von Baustein 1
    vec=1,                                 # Anzahl der Unbekannten pro Knoten (1 für Skalar)
    dim=2,                                 # Raumdimension (2 oder 3)
    ele_type=ele_type,                     # Elementtyp ('QUAD4', 'HEX8', ...)
    dirichlet_bc_info=dirichlet_bc_info    # Randbedingungen von Baustein 3
)

# Lösen
sol = solver(problem)
```

**Was macht `solver()`?**

Der Solver verwendet das **Newton-Raphson-Verfahren** – ein iteratives Verfahren für nichtlineare Gleichungssysteme. Für lineare Probleme (wie Poisson) konvergiert es in einem einzigen Schritt.

Intern passiert Folgendes:
1. JAX-FEM stellt das globale Gleichungssystem `K · u = f` auf (Steifigkeitsmatrix × Lösungsvektor = Lastvektor).
2. Die Dirichlet-Randbedingungen werden durch Zeilenelimination eingebaut.
3. Das lineare Gleichungssystem wird gelöst (standardmäßig mit JAX's BiCGSTAB-Verfahren).

**Rückgabewert:**
`sol` ist eine Liste von Arrays – ein Eintrag pro gekoppelter Unbekannter. Für ein einfaches Skalar-Problem:
```python
u = sol[0]   # Array der Form (Anzahl_Knoten, 1)
```

---

### Baustein 5: Ergebnis speichern

```python
import os
from jax_fem.utils import save_sol

data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_path = os.path.join(data_dir, 'vtk/u.vtu')

save_sol(problem.fes[0], sol[0], vtk_path)
```

`save_sol()` schreibt die Lösung im **VTK-Format** (`.vtu`) – einem offenen Standard für FEM-Ergebnisse. Diese Datei kann mit **ParaView** (kostenlos, Open Source) oder direkt in Python mit **PyVista** geladen und visualisiert werden.

---

## 6. Methode A: Python-Skript direkt ausführen

### Einfaches 2D-Beispiel (`example.py`)

Löst die Poisson-Gleichung auf einem 2D-Einheitsquadrat. Ergebnis wird als VTK gespeichert.

```bash
cd /Users/philippschafer/Projekte/JaxFEM
.venv/bin/python example.py
```

**Erwartete Ausgabe im Terminal:**
```
No. of DOFs: 1089
Solve linear system...
Wall time: 2.34 s
```

Das Ergebnis liegt danach unter `data/vtk/u.vtu`.

---

### 3D-Workflow mit PyVista-Visualisierung (`simple_workflow.py`)

Dieser Workflow löst dasselbe Problem in 3D auf einem Quader-Netz, das mit Gmsh erzeugt wird. Am Ende öffnet sich automatisch ein interaktives 3D-Fenster.

```bash
cd /Users/philippschafer/Projekte/JaxFEM
.venv/bin/python simple_workflow.py
```

**Was das Skript tut – Schritt für Schritt:**

| Schritt | Code | Was passiert? |
|---|---|---|
| 1. Netz | `box_mesh_gmsh(Nx=10, Ny=5, Nz=5, ...)` | Gmsh erstellt ein 3D-Hexaeder-Netz mit 10×5×5 Elementen |
| 2. Physik | `class Poisson(Problem)` | Poisson-Gleichung mit Gaußscher Wärmequelle in der Mitte |
| 3. RB | `location_fns = [left, right, ...]` | Alle 6 Außenflächen werden auf u = 0 gesetzt |
| 4. Lösen | `sol = solver(problem)` | Newton-Solver berechnet die Lösung |
| 5. Speichern | `save_sol(...)` | Ergebnis als `data/vtk/solution_gmsh.vtu` |
| 6. Anzeigen | `pv.Plotter()` | PyVista öffnet ein interaktives 3D-Fenster |

Im PyVista-Fenster kannst du:
- **Drehen:** Linke Maustaste gedrückt halten und ziehen
- **Zoomen:** Mausrad
- **Verschieben:** Rechte Maustaste gedrückt halten und ziehen
- **Schließen:** Fenster schließen oder `q` drücken

---

## 7. Methode B: Interaktiv mit Jupyter Notebook

Jupyter Notebook ist eine browserbasierte Oberfläche, in der du Python-Code in einzelnen Zellen schreiben und ausführen kannst. Ergebnisse (Text, Plots, Tabellen) erscheinen direkt darunter. Das ist ideal zum Experimentieren.

### Jupyter starten

```bash
cd /Users/philippschafer/Projekte/JaxFEM
.venv/bin/jupyter lab
```

Nach wenigen Sekunden öffnet sich automatisch ein Browser-Fenster mit der JupyterLab-Oberfläche.

### Das Einstiegs-Notebook öffnen

1. In der linken Dateiliste auf **`einstieg.ipynb`** doppelklicken.
2. Das Notebook öffnet sich rechts als Tabs.

### Kernel auswählen (wichtig!)

Der Kernel ist der Python-Interpreter, der im Hintergrund läuft. Es muss der richtige ausgewählt sein, damit JAX-FEM gefunden wird:

1. Oben rechts im Notebook auf den Kernel-Namen klicken (z. B. `Python 3` oder `No Kernel`).
2. In der Liste **`Python (JaxFEM)`** auswählen.
3. Bestätigen.

> Dieser Kernel ist die `.venv` des Projekts und enthält alle notwendigen Pakete.

### Zellen ausführen

Das Notebook besteht aus mehreren **Zellen**. Jede Zelle enthält entweder Erklärungstext (Markdown) oder ausführbaren Python-Code.

| Aktion | Tastenkürzel |
|---|---|
| Aktuelle Zelle ausführen und zur nächsten springen | `Shift + Enter` |
| Aktuelle Zelle ausführen und bleiben | `Ctrl + Enter` |
| Alle Zellen von oben nach unten ausführen | Menü → `Run` → `Run All Cells` |
| Kernel neu starten (bei Absturz) | Menü → `Kernel` → `Restart Kernel` |

**Reihenfolge ist wichtig:** Führe die Zellen immer von oben nach unten aus. Eine Zelle kann nur Variablen verwenden, die in vorherigen Zellen bereits definiert wurden.

### Aufbau des Einstiegs-Notebooks

Das Notebook `einstieg.ipynb` führt durch dieselben fünf Bausteine wie `example.py`, aber interaktiv mit Erklärungen und direkt im Browser angezeigtem Plot:

| Zelle | Inhalt |
|---|---|
| 1 | Pakete laden, JAX-Version anzeigen |
| 2 | `Poisson`-Klasse definieren (Physik) |
| 3 | Netz mit 32×32 Elementen erzeugen |
| 4 | Randbedingungen definieren |
| 5 | Problem lösen |
| 6 | Ergebnis als Farbplot im Notebook anzeigen |

Nach dem Ausführen aller Zellen siehst du direkt unter der letzten Zelle einen Farbplot der Poisson-Lösung – ohne externe Programme.

### Eigene Experimente

Du kannst beliebige Werte ändern und die Zellen neu ausführen:

```python
# In Zelle 3: Auflösung erhöhen
meshio_mesh = rectangle_mesh(Nx=64, Ny=64, ...)   # statt 32×32

# In Zelle 2: Wärmequelle verschieben
val = -np.array([10*np.exp(-(np.power(x[0] - 0.3, 2) + ...) / 0.02)])
#                                               ↑ war 0.5 (Mitte), jetzt 0.3 (links)
```

Führe danach **nur die geänderte Zelle und alle nachfolgenden Zellen** neu aus.

---

## 8. Ergebnisse visualisieren

### Option A: Direkt im Notebook (Matplotlib)

Das Einstiegs-Notebook zeigt bereits einen Matplotlib-Plot. Matplotlib ist für 2D-Ergebnisse gut geeignet.

```python
import matplotlib.pyplot as plt

points = meshio_mesh.points
u = sol[0].flatten()

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.tripcolor(points[:, 0], points[:, 1], u, shading='gouraud', cmap='viridis')
plt.colorbar(sc, ax=ax, label='u')
ax.set_aspect('equal')
plt.show()
```

### Option B: Interaktiv mit PyVista (3D)

PyVista ermöglicht interaktive 3D-Visualisierung und ist besonders für Volumenlösungen geeignet:

```python
import pyvista as pv

pv_mesh = pv.read('data/vtk/solution_gmsh.vtu')
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, scalars='sol', show_edges=True, cmap='plasma')
plotter.show()
```

### Option C: ParaView (empfohlen für komplexe Ergebnisse)

**ParaView** ist ein kostenloses Open-Source-Programm für professionelle wissenschaftliche Visualisierung. Es kann unter [paraview.org](https://www.paraview.org) heruntergeladen werden.

Zum Öffnen einer Ergebnisdatei:
1. ParaView starten
2. `File → Open` → `.vtu`-Datei auswählen (z. B. `data/vtk/u.vtu`)
3. Auf `Apply` klicken
4. Oben links in der Toolbar die darzustellende Variable auswählen (z. B. `sol`)

---

## 9. Unterstützte Elementtypen

JAX-FEM unterstützt verschiedene Elementtypen für 2D- und 3D-Probleme. Der Elementtyp wird beim Erstellen des Netzes und der `Problem`-Instanz angegeben.

### 2D-Elemente

| Typ | Knoten | Ordnung | Beschreibung |
|---|---|---|---|
| `QUAD4` | 4 | 1. Ordnung | Bilineares Viereck – Standard für strukturierte Netze |
| `QUAD8` | 8 | 2. Ordnung | Serendipity-Viereck – genauer, aber aufwändiger |
| `TRI3`  | 3 | 1. Ordnung | Lineares Dreieck – flexibel für unstrukturierte Netze |
| `TRI6`  | 6 | 2. Ordnung | Quadratisches Dreieck |

### 3D-Elemente

| Typ | Knoten | Ordnung | Beschreibung |
|---|---|---|---|
| `HEX8`  | 8  | 1. Ordnung | Trilinearer Hexaeder – Standard für strukturierte 3D-Netze |
| `HEX20` | 20 | 2. Ordnung | Serendipity-Hexaeder – höhere Genauigkeit |
| `TET4`  | 4  | 1. Ordnung | Lineares Tetraeder – für komplexe Geometrien (Gmsh) |
| `TET10` | 10 | 2. Ordnung | Quadratisches Tetraeder |

**Faustregel:** Beginne mit `QUAD4` (2D) oder `HEX8` (3D). Für komplexe Geometrien, die nicht in Quader/Rechtecke passen, verwende `TET4` mit Gmsh-Netz.

---

## 10. Randbedingungen verstehen

### Dirichlet-Randbedingungen (Wert fest vorgegeben)

Werden verwendet, wenn die Lösung selbst an einem Rand einen bestimmten Wert haben soll.
- **Beispiel Wärmeleitung:** Temperatur an der Wand ist T = 20°C
- **Beispiel Elastizität:** Verschiebung an der Einspannung ist u = 0

```python
# Format: [location_fns, vecs, value_fns]
dirichlet_bc_info = [
    [left, right, bottom, top],   # Wo gilt die RB?
    [0, 0, 0, 0],                  # Welche Komponente? (0=x, 1=y, 2=z)
    [zero, zero, zero, zero]       # Welcher Wert?
]
```

### Neumann-Randbedingungen (Fluss fest vorgegeben)

Werden verwendet, wenn der Gradient (Fluss, Kraft, Wärmefluss) am Rand vorgegeben ist.
- **Beispiel Wärmeleitung:** Wärmefluss q = 100 W/m² an einer Fläche
- **Beispiel Elastizität:** Zugkraft p = 10 MPa an einer Fläche

Neumann-RBs werden über die Methode `get_surface_maps()` in der Problem-Klasse definiert:

```python
class MeinProblem(Problem):
    def get_surface_maps(self):
        def neumann(u, point):
            return -np.array([100.])   # Wärmefluss = 100 W/m²
        return [neumann]
```

Zusätzlich muss beim Erstellen des Problem-Objekts `location_fns` angegeben werden:

```python
problem = MeinProblem(
    ...,
    location_fns=[top]   # Neumann-RB gilt am oberen Rand
)
```

### Robin-Randbedingungen (gemischt: Fluss abhängig vom Wert)

Für Konvektion: Der Wärmefluss hängt von der Differenz zwischen Oberflächentemperatur und Umgebungstemperatur ab.
- **Formel:** q = h · (u − u∞)
- Wird über `get_universal_kernels_surface()` definiert (fortgeschrittene Nutzung).

---

## 11. Häufige Fehler und Lösungen

### `ModuleNotFoundError: No module named 'jax_fem'`

**Ursache:** Das falsche Python wird verwendet. Die Systempython-Installation kennt JAX-FEM nicht.

**Lösung:** Immer den Interpreter aus der virtuellen Umgebung verwenden:
```bash
.venv/bin/python mein_skript.py   # richtig
python mein_skript.py             # falsch (System-Python)
```

Im Notebook: Kernel auf `Python (JaxFEM)` umstellen.

---

### `FileNotFoundError: data/vtk/...`

**Ursache:** Der Ausgabeordner existiert noch nicht.

**Lösung:** Ordner vorher anlegen:
```python
import os
os.makedirs('data/vtk', exist_ok=True)
```

---

### Das PyVista-Fenster öffnet sich nicht

**Ursache:** Auf manchen Systemen benötigt PyVista eine spezielle Rendering-Umgebung.

**Lösung:** Im Notebook kann PyVista offline gerendert werden:
```python
pv.start_xvfb()           # Nur auf Linux nötig
pv.set_jupyter_backend('static')  # Statische Bilder statt interaktivem Fenster
```

---

### Der Solver konvergiert nicht (`Newton solver did not converge`)

**Mögliche Ursachen:**
- Zu grobe Vernetzung für das Problem
- Falsche oder fehlende Randbedingungen
- Physikalisch schlecht gestelltes Problem

**Lösung:** Netz verfeinern (`Nx`, `Ny`, `Nz` erhöhen), Randbedingungen prüfen.

---

### Simulation ist sehr langsam beim ersten Aufruf

**Erwartetes Verhalten.** JAX kompiliert den Code beim ersten Aufruf zu optimiertem Maschinencode (JIT – Just-in-Time Compilation). Das dauert einige Sekunden bis Minuten. Beim zweiten Aufruf mit gleichen Dimensionen ist die Simulation deutlich schneller.

---

## 12. Ergebnis & Visualisierung – Spannungen, Dehnungen, Trajektorien

Das Notebook `experimental.ipynb` zeigt am Kragarm-Beispiel, wie man über die reinen Verschiebungen hinaus alle mechanisch relevanten Größen auswertet und visualisiert. Abschnitt 5 des Notebooks ist in fünf Unterabschnitte gegliedert.

---

### 12.1 Verschiebungen & Vergleich mit Euler-Bernoulli (5.1)

| Größe | Symbol | Einheit | Plottyp |
|---|---|---|---|
| Vertikale Durchbiegung | u_y | m | `tripcolor`, RdBu_r |
| Horizontale Verschiebung | u_x | m | `tripcolor`, viridis |

Die maximale Durchbiegung am freien Ende wird automatisch mit der **Euler-Bernoulli-Balkentheorie** verglichen:

```
w_max = F·L³ / (3·E·I)
```

Dabei gilt für einen rechteckigen Querschnitt mit Breite b = 1 m (Scheibendicke) und Höhe h:

```
I = b·h³ / 12
```

Die Abweichung zwischen FEM und analytischer Lösung liegt bei feiner Vernetzung typischerweise unter 1 %.

---

### 12.2 Spannungen (5.2)

Spannungen sind **keine** direkten Knotengrößen – JAX-FEM speichert nur Verschiebungen. Sie müssen nachträglich aus dem Verschiebungsgradienten berechnet werden.

#### Berechnungsweg

1. **Gauss-Punkt-Auswertung:** Für jedes Element werden an den 4 Gauss-Punkten (2×2-Quadratur für QUAD4) die Ansatzfunktionsableitungen ∂N/∂x im physikalischen Raum bestimmt.
2. **Verschiebungsgradient:** ∇u = (∂N/∂x)ᵀ · u_cell  →  (2×2)-Matrix
3. **Dehnung (lineare Kinematik):** ε = ½(∇u + ∇uᵀ)
4. **Spannung (Hooke'sches Gesetz):**

```
σ_xx = (λ + 2μ)·ε_xx + λ·ε_yy
σ_yy =  λ·ε_xx + (λ + 2μ)·ε_yy
τ_xy =  2μ·ε_xy
```

5. **Extrapolation auf Knoten:** Nearest-Neighbor-Zuordnung (via `scipy.spatial.cKDTree`) – jedem Knoten wird der nächstgelegene Gauss-Punkt zugewiesen.

#### Dargestellte Größen

| Größe | Symbol | Beschreibung |
|---|---|---|
| Normalspannung längs | σ_xx | Biegespannung (dominant) |
| Normalspannung quer | σ_yy | Querspannung |
| Schubspannung | τ_xy | Schubspannung in der Querschnittsebene |
| Von-Mises-Vergleichsspannung | σ_v | Plastizitätskriterium: `√(σ_xx²− σ_xx·σ_yy + σ_yy² + 3τ_xy²)` |

> **Hinweis:** Der ebene Spannungszustand (σ_zz = τ_xz = τ_yz = 0) ist für eine dünne Scheibe physikalisch korrekt. Bei einem dicken 3D-Körper wäre der ebene Verzerrungszustand zu wählen – dann ändern sich λ_eff und μ_eff.

---

### 12.3 Dehnungen (5.3)

Analog zu den Spannungen werden die Dehnungskomponenten dargestellt:

| Größe | Symbol | Beschreibung |
|---|---|---|
| Längsdehnung | ε_xx | Biegung → oben Druck, unten Zug |
| Querdehnung | ε_yy | Poissoneffekt |
| Gleitungsmaß | γ_xy / 2 = ε_xy | Schubverzerrung |
| Erste Hauptdehnung | ε₁ | `½(ε_xx+ε_yy) + √[(½(ε_xx−ε_yy))² + ε_xy²]` |

Die **Hauptdehnungen** sind die Eigenwerte des Dehnungstensors – sie zeigen die größte Längenänderung unabhängig von der Koordinatenrichtung.

---

### 12.4 Hauptspannungen & Trajektorien (5.4)

#### Hauptspannungsberechnung

Die Hauptspannungen sind die Eigenwerte des 2D-Spannungstensors:

```
σ₁,₂ = ½(σ_xx + σ_yy) ± √[ (½(σ_xx − σ_yy))² + τ_xy² ]
```

Der zugehörige Winkel der ersten Hauptspannungsrichtung:

```
θ = ½ · arctan(2τ_xy / (σ_xx − σ_yy))
```

#### Trajektorien-Plot

Das Notebook zeigt **doppelköpfige Pfeile** auf einem regulären Gitter (25×6 Punkte), interpoliert mit `scipy.interpolate.LinearNDInterpolator`:

- **Rot** → Richtung von σ₁ (erste Hauptspannung)
- **Blau** → Richtung von σ₂ (zweite Hauptspannung, senkrecht zu σ₁)
- **Grün** → σ₁-Isobare (Konturlinien gleicher erster Hauptspannung)
- **Grauer Hintergrund** → von-Mises-Vergleichsspannung

> **Physikalische Interpretation am Kragarm:**
> - An der **Einspannung** (x = 0): σ₁ verläuft fast horizontal (dominante Biegespannung), σ₂ nahezu vertikal
> - In der **Nähe der neutralen Faser** (y = h/2): Schubspannung dominiert → Hauptspannungen unter ±45°
> - Am **freien Ende** (x = L): Spannungen gering (nur Last-Einleitung)

---

### 12.5 Querschnittsplot – FEM vs. Analytik (5.5)

Der abschließende Plot schneidet den Träger an zwei Stellen (x ≈ 0 und x = L/2) auf und vergleicht die FEM-Biegespannung σ_xx mit der Euler-Bernoulli-Lösung:

```
σ_xx(y) = M(x) · (y − h/2) / I
```

mit M(x) = F·(L − x) (Biegemoment bei Einzellast am Kragarm-Ende).

Die lineare Verteilung über die Höhe ist das charakteristische Kennzeichen der Biegetheorie. Das FEM-Ergebnis sollte bei ausreichend feiner Diskretisierung (≥ 4 Elemente über die Höhe) sehr gut übereinstimmen.

---

### Warum Nearest-Neighbor statt SPR-Glättung?

Die hier verwendete einfache Nearest-Neighbor-Extrapolation ist robust und schnell, aber nicht optimal. Für höhere Genauigkeit könnte man das **Superconvergent Patch Recovery (SPR)**-Verfahren verwenden – dabei werden die Gauss-Punktwerte in einem Patch um jeden Knoten durch ein Polynom angepasst. Der Aufwand ist deutlich höher, der Gewinn bei ausreichend feinen Netzen aber gering.

---

*Zuletzt aktualisiert: April 2026*
