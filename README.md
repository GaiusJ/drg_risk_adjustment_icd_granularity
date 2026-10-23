# ICD-Granularität und Risikoadjustierung

Analyse der Kapitel- und Gruppenebene für die Vorhersage der Mortalität und Verweildauer im deutschen DRG-System

Masterarbeit im Studiengang Wirtschaftsingenieurwesen (Gesundheitstechnik), Technische Universität Berlin, 2026

Autor: Julius Bamberg

## Hintergrund

Die Risikoadjustierung wird für Qualitätsvergleiche und im morbiditätsorientierten Risikostrukturausgleich (Morbi-RSA) breit verwendet.
Diagnosen stellen dabei eine zentrale Komponente dar. Sie sind hierarchisch aufgebaut, wobei eine tiefere Granularität mehr Informationen enthält.
In dieser Arbeit wird systematisch untersucht, welchen Einfluss die Granularität von Diagnosen auf die Vorhersage der Krankenhausmortalität und der Verweildauer hat.

Die Analyse basiert auf den Ergebnissen von Stausberg und Hagn (2015). Die Ergebnisse werden bestätigt, erweitert und auf ihre zeitliche Stabilität geprüft.

## Ergebnisse

Die Gruppenebene ist der Kapitelebene durchgängig überlegen (ΔAUROC: 0,042 [Mortalität] bzw. 0,022 [Verweildauer]).
Eine Überanpassung ist für beide Ebenen vernachlässigbar.
Den größten Einfluss auf die Modellgüte haben Kovariaten (ω²: 0,436 bzw. 0,557) und Nebendiagnosen (ω²: 0,163 bzw. 0,127). Die Granularität hat im Vergleich einen kleinen Effekt (ω²: 0,033 bzw. 0,009).
Diagnosen erzielen eine höhere Modellgüte als aggregierte Komorbiditätsscores.
Die vollständigen Ergebnisse sind in der schriftlichen Ausarbeitung (Ausarbeitung/Abschlussarbeit.pdf) zu finden. Dieses wird nach der Bewertung der Arbeit veröffentlicht.

## Aufbau des Projekts

Das Projekt ist auf die Notebooks und das zugrunde liegende Programmpaket aufgeteilt.

### Programmpaket

'src/drg_risk' ist das Programmpaket. Dort befindet sich der wiederverwendbare Code zum Laden der Daten, zum Trainieren der Modelle und zur Bewertung der Ergebnisse. Dieser wird von den Notebooks aufgerufen.

### Notebooks

Die Notebooks (Notebooks/) stellen den Einstieg in das Projekt dar. Sie werden der Reihe nach ausgeführt und starten alle Schritte zentral. Dazu gehören die Datenaufbereitung, das Trainieren der Modelle und die Evaluierung dieser.

| Notebook	| Aufgabe |
| :--- | :--- |
| 01_convert_excel_icd_to_json | ICD-Katalog aus Excel in ein maschinenlesbares Format überführen |
| 02_generate_comorbidity_lookups | Zuordnung für die Komorbiditäts-Indizes (Charlson, Elixhauser) erzeugen |
| 03_data_preprocessing | Campus File bereinigen und auf Kapitel- und Gruppenebene aufbereiten |
| 04_campus_file_summary | Überblick über die Daten |
| 05_running_models | Die 240 Modellkonfigurationen trainieren und evaluieren |
| 06_Auswertung | Ergebnisse der Modelle vergleichen |
| 07_stabilitaetspruefung | Vergleich 2010 (Campus File) mit 2024 (Datenbrowser) |

### Projektstruktur

    drg_risk/
    ├── Notebooks/                                      Ablauf der Analyse (siehe oben)
    ├── src/drg_risk/
    │   ├── data/
    │   │   ├── dataloader.py                           Daten einlesen
    │   │   ├── datapreprocessor.py                     Daten bereinigen
    │   │   └── dataclasses.py                          Datenstrukturen
    │   ├── models/
    │   │   └── ordinal_log_regression_model.py         eigenes Verweildauer-Modell
    │   ├── pipeline/
    │   │   └── trainer.py                              Training und Kreuzvalidierung
    │   ├── evaluation/
    │   │   ├── evaluator.py                            Gütemaße berechnen
    │   │   ├── scores.py, mauc.py                      einzelne Metriken
    │   │   ├── score_postprocessing.py                 Gütemaße nachbearbeiten
    │   │   └── typst_report_exporter.py                Ergebnisse für die Arbeit exportieren
    │   ├── analysis/
    │   │   ├── hyperparameter_impact_analysis.py       Einfluss der Hyperparamter
    │   │   ├── stability_analyzer.py                   Stabilität über die Zeit
    │   │   ├── correlation_coefficient_calculator.py   Berechnung der Korrelationskoeffizienten im Vergleich mit Stausberg und Hagn
    │   │   └── summary.py                              statistische Zusammenfassung des Campus Files 2010 
    │   └── utils/logger.py                             Dokumentation
    ├── requirements.txt                                direkte Abhängigkeiten
    └── pyproject.toml

## Daten

Das Campus File DRG 2010 unterliegt bestimmten Nutzungsbedingungen und ist deshalb nicht in diesem Repository enthalten. Es ist für wissenschaftliche Zwecke frei verfügbar [Forschungsdatenzentrum](https://www.forschungsdatenzentrum.de/de/gesundheit/drg).
Die Daten sind als 'Daten/Campus_File_DRG_2010/Campus_File_DRG_2010_DRG_Statistik.csv' einzufügen.

Da die Daten aus dem InEK-Datenbrowser ebenfalls nicht veröffentlicht werden dürfen, sind auch diese nicht im Repository vorhanden. Sie sind frei zugänglich unter [InEK Datenbrowser](https://datenbrowser.inek.org/login). Für jede Gruppe müssen sie unter 'Daten/Datenbrowser Ergebnisse/2024 inkl. Todesfälle' eingefügt werden. Dabei ist die Gruppe die Hauptdiagnose. Dies erfolgt nach dem folgenden Muster: '{Gruppennummer} {Gruppenbezeichnung} {ggf. Todesfälle}.xlsx'. Somit ergibt sich beispielsweise folgender Dateiname: 'A00-A09 Infektiöse Darmkrankheiten Todesfälle.xlsx'

Für die Berechnung der Proxy-Komorbiditätsscores werden die Gruppen als Nebendiagnosen verwendet. Diese werden unter 'Daten/Datenbrowser Ergebnisse/2024 inkl. Todesfälle/Aufruf nach Nebendiagnose' nach der folgenden Systematik gespeichert: '{gruppennummer}.xlsx (Beispiel: A00—A09.xlsx)'.

Die Dateien, die in den Notebooks 1 und 2 generiert werden, wurden dem Repository hinzugefügt. Dadurch sind die Daten aus dem InEK-Datenbrowser lediglich für das Notebook 7 notwendig. Die Notebooks 3-6 können ohne die Daten benutzt werden. Das Campus File ist weiterhin notwendig.

## Installation

Voraussetzung:
- Python 3.14 (getestet mit 3.14.3). Download: [python.org](https://www.python.org/)
- Git. Download: [git-scm.com](https://git-scm.com/book/de/v2/Erste-Schritte-Git-installieren)

Es wird empfohlen, das Projekt mithilfe eines Editors, wie z. B. Visual Studio Code zu öffnen.

1. Projekt herunterladen und in den Ordner wechseln.

```bash
git clone https://github.com/GaiusJ/drg_risk_adjustment_icd_granularity.git
cd drg_risk_adjustment_icd_granularity
```

2. Virtuelle Umgebung anlegen

Eine virtuelle Umgebung wird verwendet, um Projekte und deren Installationen sauber voneinander zu trennen.

    python -m venv .venv

Aktivieren unter Windows:

    .venv\Scripts\activate

Aktivieren unter macOS und Linux:

    source .venv/bin/activate

3. Pakete installieren

In 'requirements.txt' sind die zu verwendenden Pakte mit der exakten Version aufgeführt, mit der diese Arbeit erstellt wurde.
Der zweite Befehl ermöglicht den Import des Ordners 'src/drg_risk' als Paket.

    pip install -r requirements.txt
    pip install -e .

4. Notebooks starten

Zunächst wird ein Notebook geöffnet und der richtige Kernel ausgewählt (der auf die virtuelle Umgebung verweist). Anschließend kann das gesamte Notebook ausgeführt werden.

## Hyperparametervariation

In dieser Arbeit werden alle möglichen Kombinationen aus sechs Hyperparametern evaluiert.

| Einstellung | Werte | Anzahl |
|---|---|---|
| `icd_structures`: Granularität der Diagnosecodes | `group` (ICD-Gruppen), `chapter` (ICD-Kapitel) | 2 |
| `output_vars`: Zielgröße | `entl_grd`, `typ_vwd` | 2 |
| `input_vars`: Art der Diagnoseinformation | `charlson`, `elixhauser`, `codes` | 3 |
| `use_sdx_`: Nebendiagnosen einbeziehen | `True`, `False` | 2 |
| `kovarianten_usage`: Zusatzmerkmale | Alter, Geschlecht, Aufnahmeanlass, Anzahl Nebendiagnosen, oder `None` (ohne Zusatzmerkmale) | 2 |
| `subsets`: Patientengruppe | alle Fälle, Notfälle `N`, keine Notfälle `E`/`V`/`G`, medizinische Fälle `M`, operative Fälle `O` | 5 |

Insgesamt ergibt dies 240 unterschiedliche Modelle, jeweils 120 für die Mortalität und die Verweildauer.

### Laufzeit und Hardware

Die Ergebnisse wurden auf einem Intel Core i5-8400 mit 16 GB Arbeitsspeicher
erzeugt. Das Training und die Evaluation aller 240 Modelle dauern dort etwa sechs
Stunden.

| Modell | Dauer pro Modell |
|---|---|
| Mortalität (logistische Regression) | 128 ± 61 Sekunden |
| Verweildauer (ordinales Modell) | 43 ± 32 Sekunden |

Die Dauer pro Modell hängt nahezu linear von der Zahl der Krankenhausfälle in der jeweiligen Patient:innengruppe ab.

### Schnelltest

Um die Funktionsweise des Modells zu überprüfen, ohne alle Modelle trainieren zu müssen, ist es möglich, die Hyperparameter-Kombinationen anzupassen.
Mit diesen Kombinationen werden die Baseline-Modelle trainiert.

```python
icd_structures = ["group", "chapter"]
output_vars = ["typ_vwd", "entl_grd"]
subsets = [None]
input_vars = ["codes"]
use_sdx_ = [True]
kovarianten_usage = [["typ_alter", "sex", "aufn_anl", "anzahl_nebendiagnosen"]]
```

Erwartete Dauer: 14 Minuten

Es muss jedoch beachtet werden, dass die darauf folgenden Auswertungsschritte (Notebook 6) einen vollständigen Durchlauf aller Modelle benötigen.

## Lizenz

Dieses Projekt ist unter der GNU General Public License v3 (GPLv3) lizenziert.