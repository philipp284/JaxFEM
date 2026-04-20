---
title: "JaxFEM — GPU-beschleunigter FEM-Solver"
type: project
scope: development
status: active
last_updated: "2026-04-19"
technology: [Python, JAX, FEM, NumPy, PETSc, Automatische-Differenzierung]
domain: [Scientific-Computing, Strukturmechanik, Topologie-Optimierung]
institution: Hochschule Mainz
related:
  - path: /Users/philippschafer/Projekte/CLAUDE.md
    label: Workspace-Master
  - path: /Users/philippschafer/Projekte/TM_tools/CLAUDE.md
    label: TM_tools (Statik-Web)
  - path: /Users/philippschafer/Library/CloudStorage/OneDrive-HochschuleMainz-UniversityofAppliedSciences/03_Tragwerkslabor/06_Projekte/2026_03_11_MIB/CLAUDE.md
    label: MIB-Projekt
---

# JaxFEM

Python-Bibliothek für die Finite-Elemente-Methode (FEM), basierend auf **JAX** für automatische Differenzierung und GPU-Beschleunigung. Akademisches Projekt mit vollständiger Dokumentation.

## Zweck

- Differenzierbarer FEM-Solver für Forschung und Lehre
- GPU-Beschleunigung über JAX/XLA
- Topologie-Optimierung und nichtlineare Analyse
- Integration mit PETSc für Hochleistungsrechner

## Schlüsselpfade

```
JaxFEM/
├── jax_fem/
│   ├── solver.py          # Hauptlöser (lineare + nichtlineare Gleichungssysteme)
│   ├── basis.py           # Formfunktionen, Gauss-Integration
│   ├── problem.py         # Problemdefinition, Randbedingungen
│   ├── generate_mesh.py   # Mesh-Erzeugung
│   └── physics/           # Materialmodelle (Elastizität, Plastizität)
├── demos/                 # Beispielprobleme
├── docs/                  # Dokumentation
├── tests/                 # Testfälle
└── requirements.txt
```

## Technologie-Stack

| Komponente | Technologie |
|-----------|-------------|
| Sprache | Python 3.x |
| Kern-Framework | JAX (Google) |
| Lineare Algebra | NumPy, SciPy |
| HPC-Interface | PETSc |
| Auto-Diff | JAX grad/jit/vmap |

## Fähigkeiten

- **Elementtypen:** 2D (Dreieck, Viereck) und 3D (Tetraeder, Hexaeder)
- **Analysetypen:** Linear-elastisch, nichtlinear, Plastizität, Multi-Physik
- **Optimierung:** Topologie-Optimierung (SIMP-Methode)
- **Parallelisierung:** JIT-Kompilierung und GPU-Beschleunigung via JAX

## Verbindungen im Wissensgraph

- Verwandt mit **TM_tools** (Web-Interface für Tragwerksberechnungen)
- Relevant für **Tragwerkslabor / MIB** (Materialprüfung und Simulation)
