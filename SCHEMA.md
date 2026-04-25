# SCHEMA.md: Low-Code Workflow für Strukturanalysen

Dieses Dokument beschreibt den Prozessablauf in der Low-Code Umgebung zur automatisierten Statikberechnung basierend auf IFC-Daten.

## Prozess-Übersicht

Der Workflow ist in vier Hauptknoten unterteilt, die von der Datenaufnahme bis zur globalen Analyse führen, gefolgt von einer detaillierten Einzelpositionsprüfung.

---

### 1. IFC Parsing

- **Funktion:** Einlesen und Analysieren der BIM-Modell-Daten.
- **Input:** Rohdaten im .ifc Format.
- **Output:** Strukturierte Objektlisten (Träger, Stützen, Platten) und Materialeigenschaften.

### 2. Mich. Generierung (Modell-Generierung)

- **Funktion:** Erstellung des mechanischen Analyse-Modells.
- **Beschreibung:** Umwandlung der physischen Geometrie in ein statisches System (Achsenmodell).
- **Prozess:** Definition von Knotenpunkten und Stabelementen.

### 3. Loading and Supports

- **Funktion:** Definition der Randbedingungen und Einwirkungen.
- **Parameter:**
  - **Supports:** Festlegung der Auflagerbedingungen (starr, gelenkig, elastisch).
  - **Loading:** Aufbringen von Lastfällen (Eigengewicht, Nutzlast, Wind, Schnee).

### 4. Wegen Gesamtstatik (Global-Analyse)

- **Funktion:** Berechnung des gesamten Tragwerks als System.
- **Ziel:** Ermittlung der globalen Verformungen, Eigenfrequenzen und der Lastabtragung im Gesamtsystem.

---

## Detaillierung: Einzel-Positionen

Nach der globalen Analyse verzweigt sich das Schema in die detaillierte Bauteilprüfung:

### A. Auswahl & Filterung

- **Einzel. Positionen:** Gruppierung identischer Bauteile aus dem Gesamtmodell.
- **Fett / Representative Parts:** Identifikation der maßgebenden ("fettesten" oder am stärksten belasteten) Bauteile pro Gruppe zur effizienten Bemessung.

### B. Calculate Position (Einzelnachweis)

In diesem Schritt erfolgen die spezifischen Nachweise für jedes repräsentative Element:

1. **Analyse Component Stresses:** Prüfung der Schnittkräfte (M, V, N) auf Bauteilebene.
2. **Verify Deflection:** Kontrolle der Durchbiegung im Gebrauchszustand (GZT/GZG).
3. **Calculate Connections:** Detailnachweis der Anschlüsse und Verbindungen.
4. **Identify Critical Elements:** Markierung von Bauteilen, die Grenzwerte überschreiten.

### C. Output & Status

- **Update Element Status:** Vergabe eines Status-Flags:
  - ✅ **Approved:** Bauteil erfüllt alle Nachweise.
  - ⚠️ **Check Needed:** Manuelle Überprüfung oder Anpassung erforderlich.

### D. Iterationsschleife

- **Position Optimization Loop:** Bei Bedarf Rückführung der optimierten Querschnitte in die **Modell-Generierung (Knoten 2)**, um die Gesamtstatik mit neuen Steifigkeiten neu zu berechnen.


# Der zentrale Design-Move:

**IFC-GUID ist der Primärschlüssel durch die gesamte Pipeline.** Jeder Service nimmt GUIDs rein, gibt GUIDs raus, und der Mesh-Layer hält das kritische Mapping `IFC-GUID ↔ Mesh-Group/Elemente` persistent. Damit wird jede Stufe unabhängig austauschbar (Code_Aster heute, JAX-FEM morgen), und die Ergebnisse sind durchgängig semantisch an das BIM-Modell rückgebunden.

V

visualize

V

visualize** show_widget**

<iframe title="visualize: Ifc structural pipeline architecture" sandbox="allow-scripts allow-same-origin allow-forms" allow="fullscreen *; clipboard-write *" src="https://512b1ba895aa423c5c9173862fb8bf8b.claudemcpcontent.com/mcp_apps?connect-src=https%3A%2F%2Fesm.sh+https%3A%2F%2Fcdnjs.cloudflare.com+https%3A%2F%2Fcdn.jsdelivr.net+https%3A%2F%2Funpkg.com&resource-src=https%3A%2F%2Fesm.sh+https%3A%2F%2Fcdnjs.cloudflare.com+https%3A%2F%2Fcdn.jsdelivr.net+https%3A%2F%2Funpkg.com+https%3A%2F%2Fassets.claude.ai&dev=true"></iframe>

#### Endpoint-Contract

Jeder Service ist ein FastAPI-Notebook oder Python-Microservice. Sie sind stateless bis auf die Referenz zum GUID-Store. Die `parse_id` bzw. `result_id` sind nur Handles auf Artefakte im Store:

```
POST /ifc/parse
  in:  { file_id | ifc_url }
  out: { parse_id,
         elements: [{ guid, ifc_type, geometry_ref, material_ref,
                      is_load_bearing: bool }, ...] }

POST /supports/{parse_id}
  in:  { rules: [{ filter: "IfcSlab on ground", type: "fixed|pinned|roller" }] }
  out: { supports: [{ guid, type, location, dof_released }] }

POST /loads/{parse_id}
  in:  { load_cases: [{ name: "LC1_dead", type: "self_weight", gamma_G: 1.35 },
                     { name: "LC2_live", type: "area_load",
                       magnitude: 3.0, unit: "kN/m^2",
                       target_guids: [...] }] }
  out: { loads: [...] }

POST /mesh/{parse_id}
  in:  { element_size: 0.25, target_family: "BEAM|SHELL|SOLID" }
  out: { mesh_id,
         guid_map: { "0abC...": { mesh_group: "bm_042", node_ids: [..] } } }

POST /fem/{mesh_id}
  in:  { analysis: "linear_static",
         load_combinations: ["1.35*LC1 + 1.5*LC2"] }
  out: { result_id, status, med_file_url, log_url }

GET /results/{result_id}/element/{guid}
GET /results/{result_id}/element/{guid}/top3
GET /results/{result_id}/report/statik
```

#### Konsistentes GUID-Schema

Die Results-API liefert für jedes tragende Element eine parametrisierte Schnittgrößenkurve (für Stabwerk-Elemente: `s ∈ [0, 1]` entlang der Achse; für Schalen: Knoten-ID plus lokale Koordinaten). Beispiel-Payload:

json

```json
{
"ifcGlobalId":"0abC3fDe7Hj_XyZ92",
"ifc_type":"IfcBeam",
"length_m":6.40,
"load_combination":"1.35*LC1 + 1.5*LC2",
"top3":[
{"s":0.50,"x_m":3.20,
"sigma_v":152.3,"sigma_unit":"N/mm^2",
"M_y":-184.5,"M_z":12.1,"N":-45.2,
"V_y":0.8,"V_z":22.1,"T":0.0,
"critical_for":"Biegung My"},
{"s":0.00,"x_m":0.00,
"sigma_v":98.7,"M_y":0,"V_z":72.5,
"critical_for":"Querkraft Vz (Auflager)"},
{"s":1.00,"x_m":6.40,
"sigma_v":94.1,"M_y":0,"V_z":-72.5,
"critical_for":"Querkraft Vz (Auflager)"}
]
}
```

Die "Top-3"-Logik ist ein einfacher Post-Processor über das volle Ergebnisfeld: pro Element die drei Positionen mit höchstem `sigma_v` (Vergleichsspannung nach v. Mises) — optional filterbar auf "höchstes M", "höchstes N", "höchstes V". Diese Auswahl ist der Backbone für den automatisch generierten statischen Nachweis: pro Element drei Bemessungsstellen → drei Nachweise → Statik-Dokument.

#### Low-Code Ebene

Für den Node-RED-Charakter konkret drei Kandidaten, in Reihenfolge der Empfehlung für Deinen Stack:

**n8n** ist der beste Match. Modernes Node-RED, selbst hostbar (Docker), HTTP-Request-Nodes nativ, hat Schleifen und Branches visuell, kann Python-Code-Nodes ausführen, speichert Flows als JSON (versionierbar). Der "Execute Workflow"-Knoten erlaubt Dir die Pipeline in Sub-Flows zu zerlegen (genau das, was Du für IFC→BC→Mesh→Solve→Results brauchst). Die UI ist deutlich aufgeräumter als Node-RED.

**Node-RED** klassisch. Reifer, riesiges Palette-Ökosystem, aber UI altbacken und die JavaScript-Fokussierung passt schlechter zu Deinem Python-Stack. Sinnvoll, wenn Du den Flow direkt im Browser für Studierende sichtbar machen willst — das hat einen starken didaktischen Effekt.

**Prefect** (oder Dagster) ist kein Low-Code, aber Python-nativ und DAG-basiert. Wenn Du später feststellst, dass die visuelle Ebene nur beim Prototyping hilft und Du produktiv DAGs in Code willst — das ist dann der Exit.

Für den Einstieg würde ich n8n in einem Docker-Container neben den FastAPI-Services deployen. Jeder Service bekommt einen HTTP-Request-Knoten als Node in n8n, die Pipeline wird visuell gezeichnet, und der Output des letzten Knotens ist der Trigger für die Statik-Dokumentgenerierung (z. B. via python-docx aus Deinem Stack).

---

Wenn Du willst, baue ich Dir als nächsten Schritt eines von drei Dingen:

1. Ein lauffähiges **FastAPI-Skeleton** für `/ifc/parse` mit IfcOpenShell (inkl. Erkennung tragender Elemente via `PredefinedType` und `LoadBearing`-Property)
2. Einen  **n8n-Workflow als JSON-Export** , der die 7-Stufen-Pipeline orchestriert — Du kannst ihn direkt importieren
3. Das **RDF-Schema** für den GUID-Store (wie werden Supports, Loads, Results als Triples im BOT/MiB-Namespace modelliert) — das wäre die semantische Grundlage, die das Ganze erst Engineer-OS-fähig macht

---

*Dokumentation erstellt für die Low-Code Statik-Umgebung.*
