from drg_risk.data.datapreprocessor import DataPreprocessor
from drg_risk import DATA_DIR, REPORT_DIR
from drg_risk.utils.logger import Logger

import os
import numpy as np
import pandas as pd
import json

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import wasserstein_distance
import functools
from scipy import stats


root_path = os.path.join(DATA_DIR, "Datenbrowser Ergebnisse", "2024 inkl. Todesfälle")


class StabilityAnalyzer():
    """
    Klasse zum Berechnen der zeitlichen Stabilität zwischen dem Campus File 2010 und den Daten des InEK-Datenbrowser 2024.
    Berechnet die prozentuale Veränderung der Fallzahl, Mortalitätsrate, die Wasserstein-Distanz für die Verweildauer und das Alter, sowie den Spearman-Rangkorrelationskoeffizienten für die Nebendiagnosen.
    """
    def __init__(self, stichprobe_campus_file = 0.1):
        """
        Args:
            stichprobe_campus_file: Stichprobenfaktor des Campus Files (Standard 0.1 = 10 %), wird zur Hochrechnung auf Grundgesamtheit verwendet.
        """
        self.stichprobe_campus_file = stichprobe_campus_file
        self.logger = Logger("stability_analyzer")
        self._load_data()

    def _load_data(self):
        """
        Lädt Campus File (2010) via DataPreprocessor und InEK-Datenbrowser-Ergebnisse (2024), harmonisiert beide Datenquellen und speichert sie in self.analysis_df als MultiIndex-DataFrame.
        """
        self.logger.info("Daten werden geladen.")
        preprocessor = DataPreprocessor(
            data_path =                 os.path.join(DATA_DIR, "Campus_File_DRG_2010", "Campus_File_DRG_2010_DRG_Statistik.csv"),
            icd_sys_path =              os.path.join(DATA_DIR, "icd_systematik_2010.json"),
            ops_sys_path =              os.path.join(DATA_DIR, "ops_systematik_2010.json"),
            columns_metadata_path =     os.path.join(DATA_DIR, "Campus_File_DRG_2010", "columns_metadata.json"),
            subgroups_path =            os.path.join(DATA_DIR, "icd_subgruppen.json"),
            drg_path =                  os.path.join(DATA_DIR, "Fallpauschalenkatalog_2010.xls"),
        )
        preprocessor.load_ungrouped_dataframe()

        self.icd_systematik = preprocessor.icd_systematik

        # Dictionary für schnelles Auslesen des Kapitels für jede Gruppe
        self.chapter_lookup = {}
        for _, chapter in self.icd_systematik.items():
            for _, group in chapter["groups"].items():
                self.chapter_lookup[group["number"]] = f"{chapter["number"]}: {chapter["name"]}"

        self.logger.info("Campus File geladen.")
        self.logger.info("Verarbeite Campus File weiter.")

        self.df = preprocessor.df

        self.df_grouped = self.df.groupby("icd_hd3")

        icd_cols = [col for col in self.df.columns if col.startswith('icd_nd')]
        ops_cols = [col for col in self.df.columns if col.startswith('ops_ko')]

        with open(os.path.join(DATA_DIR, "icd_systematik_2010.json"), "r") as f:
            self.icd_sys = json.load(f)

        # Ergebnisse des Campus Files in finales DataFrame einfügen
        # Platzhalter für Datenbrowser-Werte, werden weiter unten nach dem Laden befüllt
        rows = []
        for name, group_df in self.df_grouped:
            new_row = {
                ("icd_group", "", ""): name,
                ("Fallzahl", "", "campus_file_2010"): len(group_df) / self.stichprobe_campus_file,
                ("Fallzahl", "", "datenbrowser_2024"): None,
                ("Todesfälle", "", "campus_file_2010"): sum(group_df["entl_grd"] == 5) / self.stichprobe_campus_file,
                ("Todesfälle", "", "datenbrowser_2024"): None,
                ("Geschlecht", "weiblich", "campus_file_2010"): sum(group_df["sex"] == "w") / group_df.shape[0],
                ("Geschlecht", "männlich", "campus_file_2010"): sum(group_df["sex"] == "m") / group_df.shape[0],
                ("Geschlecht", "weiblich", "datenbrowser_2024"): None,
                ("Geschlecht", "männlich", "datenbrowser_2024"): None,
                ("Verweildauer", "", "campus_file_2010"): self._return_processed_dict_campus_file(group_df, ["typ_vwd"], sort=False),
                ("Verweildauer", "", "datenbrowser_2024"): None,
                ("Alter", "", "campus_file_2010"): self._return_processed_dict_campus_file(group_df, ["typ_alter"], sort=False),
                ("Alter", "", "datenbrowser_2024"): None,
                ("Nebendiagnosen", "", "campus_file_2010"): self._return_processed_dict_campus_file(group_df, icd_cols),
                ("Nebendiagnosen", "", "datenbrowser_2024"): None,
                ("Prozeduren", "", "campus_file_2010"): self._return_processed_dict_campus_file(group_df, ops_cols),
                ("Prozeduren", "", "datenbrowser_2024"): None,
                ("DRG", "", "campus_file_2010"): self._return_processed_dict_campus_file(group_df, ["drgh"], trunc = True),
                ("DRG", "", "datenbrowser_2024"): None,
            }

            rows.append(new_row)

        self.analysis_df = pd.DataFrame.from_records(rows)
        self.analysis_df.columns = pd.MultiIndex.from_tuples(self.analysis_df.columns)
        self.analysis_df = self.analysis_df.set_index("icd_group")
        self.icd_groups = self.analysis_df.index

        self._convert_icd_ops_to_groups()

        self.logger.info("Campus File weiterverarbeitet.")
        self.logger.info("Lade Datenbrowser Ergebnisse.")

        self.df_2024 = self._return_processed_dict_datenbrowser(self.icd_groups, root_path)

        self.logger.info("Datenbrowser Ergebnisse geladen.")

        # Ergebnisse des InEK-Datenbrowsers in DataFrame einfügen
        self.analysis_df[("Fallzahl", "", "datenbrowser_2024")] = self.df_2024["Fallzahl"]
        self.analysis_df[("Todesfälle", "", "datenbrowser_2024")] = self.df_2024["Todesfälle"]
        self.analysis_df[("Geschlecht", "weiblich", "datenbrowser_2024")] = self.df_2024[("Geschlecht", "weiblich")]
        self.analysis_df[("Geschlecht", "männlich", "datenbrowser_2024")] = self.df_2024[("Geschlecht", "männlich")]
        self.analysis_df[("Verweildauer", "", "datenbrowser_2024")] = self.df_2024["Verweildauer"]
        self.analysis_df[("Alter", "", "datenbrowser_2024")] = self.df_2024["Alter"]
        self.analysis_df[("Nebendiagnosen", "", "datenbrowser_2024")] = self.df_2024["Nebendiagnosen"]
        self.analysis_df[("Prozeduren", "", "datenbrowser_2024")] = self.df_2024["Prozeduren"]
        self.analysis_df[("DRG", "", "datenbrowser_2024")] = self.df_2024["DRG"]

        self.logger.info("Datenquellen zusammengeführt.")

    def _return_processed_dict_campus_file(self, group_df, cols, sort = True, trunc = False):
        """
        Aggregiert Spalten (ICD-Nebendiagnosen, OPS-Prozeduren o.ä.) einer ICD-Gruppe zu einem Häufigkeitsdictionary. Hochrechnung auf Grundgesamtheit via stichprobe_campus_file.

        Args:
            group_df: Teilmenge des Campus Files für eine ICD-Gruppe
            cols: Spaltennamen die aggregiert werden sollen
            sort: Wenn True, absteigend nach Häufigkeit sortiert
            trunc: Wenn True, ICD-Codes auf dreistellige Basis-ICD gekürzt
        Returns:
            Dict {Code: Fallzahl}
        """
        flattend_array_icd = group_df[cols].to_numpy().ravel()
        flattend_array_icd = flattend_array_icd[pd.notna(flattend_array_icd)]

        # relative Häufigkeit berechnen
        names, counts = np.unique(flattend_array_icd, return_counts=True, sorted=sort)
        counts = counts / self.stichprobe_campus_file

        # ggf. Kürzen der ICD-Codes
        if trunc:
            new_names = [name[:3] for name in names]
            new_counts = dict.fromkeys(new_names, 0)
            for index, name in enumerate(new_names):
                    new_counts[name] += counts[index]
            counts = new_counts
            names = new_names
        if sort:
            return pd.Series(counts, index = names).sort_values(ascending=False).to_dict()
        else:
            return pd.Series(counts, index = names).to_dict()
        
    def _convert_icd_ops_to_groups(self):
        """
        Befüllt self.icd_code_to_group und self.ops_code_to_group als Lookup-Dictionaries für die Zuordnung von Einzelcodes zur übergeordneten Gruppe laut ICD/OPS-Systematik 2010.
        """

        # Systematiken laden
        with open(os.path.join(DATA_DIR, "icd_systematik_2010.json"), "r", encoding="utf-8") as f:
            icd_systematik =  json.load(f)
        with open(os.path.join(DATA_DIR, "ops_systematik_2010.json"), "r", encoding="utf-8") as f:
            ops_systematik =  json.load(f)

        # Über Systematiken iterieren und Dictionaries befüllen
        self.icd_code_to_group: dict[str, str] = {}
        for chapter in icd_systematik.values():
            for group in chapter["groups"].values():
                for icd_code in group["codes"]:
                    self.icd_code_to_group[icd_code["code"]] = group["number"]
        self.ops_code_to_group: dict[int, str] = {}
        for chapter in ops_systematik.values():
            for group in chapter["groups"].values():
                start, end = group["number_start"], group["number_end"]
                for code_int in range(start, end + 1):
                    self.ops_code_to_group[code_int] = f"{start}-{end}"

    def _get_icd_group_from_code(self, codes: list[str]) -> list[str]:
        """
        Gibt ICD-Gruppe des gegebenen Codes zurück.
        Codes mit führendem '-' oder ohne Eintrag in der Systematik werden übersprungen.
        """
        return [
            self.icd_code_to_group[code]
            for code in codes
            if not code.startswith("-") and code in self.icd_code_to_group
        ]

    def _get_ops_group_from_code(self, codes: list[str]) -> list[str]:
        """
        Gibt OPS-Gruppe des gegebenen Codes zurück.
        Codes mit führendem '-' oder ohne Eintrag in der Systematik werden übersprungen.
        """
        result = []
        for code in codes:
            if code.startswith("-"):
                continue
            code_int = int(code.replace("-", "")[:3])
            if code_int in self.ops_code_to_group:
                result.append(self.ops_code_to_group[code_int])
        return result
    
    def _clean_drg_to_3_digits(self, code):
        """
        Bereinigt DRG-Code und kürzt auf dreistelligen Basis-DRG. Mängel oder falsche Typen werden zu np.nan.
        """
        if isinstance(code, str):
            code = code.strip()
            if len(code) == 4 and code[0].isalpha():
                return code[0:3]
        return np.nan
    
    def _process_single_group(self, icd_group: str, root_path: str) -> dict:
        """
        Liest Excel-Exportdateien des InEK-Datenbrowsers für eine ICD-Gruppe ein und extrahiert Fallzahl, Mortalität, Geschlecht, Verweildauer, Alter, Nebendiagnosen, Prozeduren und DRG-Verteilung.
        Wenn keine Datei vorhanden: Fallzahl = 0, alle anderen Felder None.

        Args:
            icd_group: ICD-Gruppencode (z.B. 'A00-A09')
            root_path: Pfad zum Ordner mit den Datenbrowser-Exceldateien
        Returns:
            Dict mit allen extrahierten Kennzahlen
        """
        paths = [p for p in os.listdir(root_path) if p.startswith(icd_group)]
        
        try:
            path_full = next(p for p in paths if "Todesfälle" not in p)

            # Ergebnisse des InEK-Datenbrowsers laden
            excel_full = pd.read_excel(os.path.join(root_path, path_full), sheet_name=None)
            info = excel_full["Info"]["Unnamed: 1"]

            # Daten extrahieren
            fallzahl   = info.iloc[9]
            anteil_m   = info.iloc[12]
            anteil_w   = info.iloc[13]
            mean_vwd   = excel_full["Info"]["Unnamed: 4"].iloc[13]

            alter_df   = excel_full["Info"][["Unnamed: 9", "Unnamed: 10"]].iloc[9:25]
            alter      = dict(zip(alter_df["Unnamed: 9"], alter_df["Unnamed: 10"]))

            nd_df      = excel_full["Nebendiagnosen"][["Kode", "Fälle"]]

            # Nebendiagnosen auf Gruppen-Ebene zusammenfassen
            nd_groups  = self._get_icd_group_from_code(nd_df["Kode"].tolist())
            nebendiagnosen = {}
            for g, v in zip(nd_groups, nd_df["Fälle"]):
                nebendiagnosen[g] = nebendiagnosen.get(g, 0) + v

            # Prozeduren auf Gruppen-Ebene zusammenfassen
            ops_df     = excel_full["Prozeduren"][["Kode", "Fälle"]]
            ops_groups = self._get_ops_group_from_code(ops_df["Kode"].tolist())
            prozeduren = {}
            for g, v in zip(ops_groups, ops_df["Fälle"]):
                prozeduren[g] = prozeduren.get(g, 0) + v

            # DRG aufbereiten und auf Basis-DRG-Ebene zusammenfassen
            drg_df = excel_full["DRG-Verteilung"][["DRG", "Fälle"]]
            drg_groups =  drg_df['DRG'].apply(self._clean_drg_to_3_digits).dropna()
            drg = {}
            for d, v in zip(drg_groups, drg_df["Fälle"]):
                drg[d] = drg.get(d, 0) + v

        # keine weiteren Daten vorhanden
        except StopIteration:
            fallzahl, anteil_m, anteil_w = 0, None, None
            mean_vwd, alter, nebendiagnosen, prozeduren, drg = None, None, None, None, None

        try:
            path_death = next(p for p in paths if "Todesfälle" in p)
            excel_death = pd.read_excel(os.path.join(root_path, path_death), sheet_name=None)
            todesfälle = excel_death["Info"]["Unnamed: 1"].iloc[9]
        except StopIteration:
            todesfälle = 0

        return {
            "icd_group":    icd_group,
            "fallzahl":     fallzahl,
            "todesfälle":   todesfälle,
            "anteil_w":     anteil_w,
            "anteil_m":     anteil_m,
            "mean_vwd":     mean_vwd,
            "alter":        alter,
            "nebendiagnosen": nebendiagnosen,
            "prozeduren":   prozeduren,
            "drg":          drg,
        }

    def _return_processed_dict_datenbrowser(self, icd_groups: list[str], root_path: str) -> pd.DataFrame:
        """
        Parallele Verarbeitung von mehreren ICD-Gruppen.
        Berechnet demografische Kennzahlen und strukturiert Ergebnisse in DataFrame.
        ThreadPoolExecutor mit 8 Workern.
        """
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(
                functools.partial(self._process_single_group, root_path=root_path),
                icd_groups
            ))

        rows = []
        for r in results:
            rows.append({
                ("icd_group", ""):              r["icd_group"],
                ("Fallzahl", ""):               r["fallzahl"],
                ("Todesfälle", ""):             r["todesfälle"],
                ("Geschlecht", "weiblich"):     r["anteil_w"],
                ("Geschlecht", "männlich"):     r["anteil_m"],
                ("Verweildauer", ""):           r["mean_vwd"],
                ("Alter", ""):                  r["alter"],
                ("Nebendiagnosen", ""):         r["nebendiagnosen"],
                ("Prozeduren", ""):             r["prozeduren"],
                ("DRG", ""):                    r["drg"],
            })

        df = pd.DataFrame.from_records(rows)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df.set_index(("icd_group", ""))

    def _to_typst_formatted(self, df):
        """
        Formatiert ein DataFrame zu einer Typst Tabelle.
        """
        df = df.reset_index()
        numeric_cols = df.select_dtypes(include=['number']).columns
        typst_table = df.style.format(
            precision=3,
            subset=numeric_cols,
            decimal=",",
            thousands="_"
        ).hide(axis="index").to_typst().replace("$", "")

        num_cols = len(df.columns) 
        # leere Zellen entfernen, wegen Index
        fixed_typst = typst_table.replace("[], ", "")

        auto_align_column_string = f"columns: ({', '.join(['auto'] * num_cols)}),"
        # 1. Spalte linksbündig, die andere rechtsbündig
        table_align_string = "  align: (col, row) => if col == 0 { left } else { right },"

        lines = fixed_typst.split("\n")

        new_lines = []

        for line in lines:
            if "columns:" in line or "table(" in line:
                continue
            new_lines.append(line)

        # Tabelle erstellen
        typst_table = (
            f"#table(\n"
            f"  {auto_align_column_string}\n"
            f"{table_align_string}\n"
            + "\n".join(new_lines)
        )

        return typst_table

    def check_stability(self, icd_groups = None, icd_chapters = None):
        """
        Erstellt einen Bericht zur Stabilität der gegebenen ICD-Gruppe.
        Wenn keine ICD-Gruppe verwendet wird, werden alle ICD-Gruppen verwendet.
        Speichert den Bericht als typ Datei.
        """
        self.logger.info("Starte Stabilitätsanalyse.")
        original = self.analysis_df.copy()

        # Daten filtern auf richtige ICD-Gruppe
        if icd_chapters is not None:
            icd_chapters_str = [str(c) for c in icd_chapters]
            all_group_numbers = []
            for key_c, value_c in self.icd_sys.items():
                chapter_number = str(value_c["number"])
                if chapter_number not in icd_chapters_str:
                    continue
                groups = value_c["groups"]
                group_numbers = [group["number"] for key, group in groups.items()]
                all_group_numbers.extend(group_numbers)
            self.analysis_df = self.analysis_df[
                self.analysis_df.index.isin(all_group_numbers)
            ]
        if icd_groups is not None:
            self.analysis_df = self.analysis_df[
                self.analysis_df.index.isin(icd_groups)
            ]

        self.logger.info("Berechne prozentuale Veränderungen.")
        (mean_count, std_count), änderung_fallzahl, änderung_fallzahl_chapter = self._check_change_in_counts()
        (mean_deaths, std_deaths), änderung_mortalitätsrate, änderung_mortalitätsrate_chapter = self._check_change_in_deaths()

        self.logger.info("Berechne Änderung der Geschlechterverteilung.")
        (mean_change_w, std_change_w), (mean_change_m, std_change_m), = self._check_change_in_sex()

        self.logger.info("Berechne Änderung der Verweildauer.")
        wd_los, änderung_verweildauer, änderung_verweildauer_chapter = self._check_change_in_los()

        self.logger.info("Berechne Änderung der Altersstruktur.")
        (mean_wd_age, std_wd_age), änderung_altersverteilung, änderung_altersverteilung_chapter = self._check_change_in_age()

        self.logger.info("Berechne Änderungen im Kodierverhalten.")
        (mean_spear_nd, std_spear_nd), jaccard_nd, änderung_nd, änderung_nd_chapter = self._check_change_in_nd()

        self.logger.info("Erstelle Report.")

        os.makedirs(REPORT_DIR, exist_ok=True)
        label = f"_chapter_{''.join(icd_chapters)}" if icd_chapters else ""
        label += f"_groups_{'_'.join(icd_groups)}" if icd_groups else ""
        os.makedirs(os.path.join(REPORT_DIR, "stability_reports"), exist_ok=True)
        path = os.path.join(REPORT_DIR, "stability_reports", f"stability_analysis_report{label}.typ")

        # Bericht schreiben
        subtitle = ""
        if icd_chapters:
            subtitle += f"Chapter: {', '.join(icd_chapters)}"
        if icd_groups:
            subtitle += f"Gruppen: {', '.join(icd_groups)}"

        # Typst-Dokument mit Stabilitätskennzahlen und Tabellen für alle Dimensionen

        # Formatiert Zahlen im deutschen Format (Punkt als Tausendertrennzeichen -> Punkt, Dezimalpunkt -> Komma)
        auswertungsstring = f"""
        #show regex("[\d_]{{2,}}_[\d_]+|\d+\.\d+|(\d+)_(\d+,\d+)"): it => {{
            if "_" in it.text {{
                if "," in it.text {{
                it.text.replace("_", ".")
                }} else {{
                it.text.replace("_", ".")
                }}
            }} else {{
                it.text.replace(".", ",")
            }}
        }}

        == Stabilitätsanalyse {subtitle}

        === Fallzahl

        Mittelwert Änderung: {mean_count:.3} %; Standardabweichung: {std_count:.3} %

        Stabilste Gruppen:
        {self._to_typst_formatted(
            änderung_fallzahl.sort_values(by = "Änderung [Prozent]", key = abs, ascending = True).head(5)
        )}

        Instabilste Gruppen:
        {self._to_typst_formatted(
            änderung_fallzahl.sort_values(by = "Änderung [Prozent]", key = abs, ascending = False).head(5)
        )}

        Stabilste Kapitel:
        {self._to_typst_formatted(
            änderung_fallzahl_chapter.sort_values(by = "Änderung [Prozent]", key = abs, ascending = True).head(5)
        )}

        Instabilste Kapitel:
        {self._to_typst_formatted(
            änderung_fallzahl_chapter.sort_values(by = "Änderung [Prozent]", key = abs, ascending = False).head(5)
        )}

        === Mortalität

        Mittelwert Änderung: {mean_deaths:.3} Prozentpunkte; Standardabweichung: {std_deaths:.3} Prozentpunkte

        Stabilste Gruppen:
        {self._to_typst_formatted(
            änderung_mortalitätsrate.sort_values(by = "Änderung [Prozentpunkte]", key = abs, ascending = True).head(5)
        )}

        Instabilste Gruppen:
        {self._to_typst_formatted(
            änderung_mortalitätsrate.sort_values(by = "Änderung [Prozentpunkte]", key = abs, ascending = False).head(5)
        )}

        Stabilste Kapitel:
        {self._to_typst_formatted(
            änderung_mortalitätsrate_chapter.sort_values(by = "Änderung [Prozentpunkte]", key = abs, ascending = True).head(5)
        )}

        Instabilste Kapitel:
        {self._to_typst_formatted(
            änderung_mortalitätsrate_chapter.sort_values(by = "Änderung [Prozentpunkte]", key = abs, ascending = False).head(5)
        )}

        === Geschlechterverteilung

        Mittelwert Änderung Anteil weiblich: {mean_change_w:.3} Prozentpunkte; Standardabweichung: {std_change_w:.3} Prozentpunkte

        Mittelwert Änderung Anteil männlich: {mean_change_m:.3} Prozentpunkte; Standardabweichung: {std_change_m:.3} Prozentpunkte

        === Verweildauer

        Wasserstein Distanz Änderung: {wd_los:.3} Tage

        Stabilste Gruppen:
        {self._to_typst_formatted(
            änderung_verweildauer.sort_values(by = ("Wasserstein Distanz", "", ""), key = abs, ascending = True).head(5)
        )}

        Instabilste Gruppen:
        {self._to_typst_formatted(
            änderung_verweildauer.sort_values(by = ("Wasserstein Distanz", "", ""), key = abs, ascending = False).head(5)
        )}

        Stabilste Kapitel:
        {self._to_typst_formatted(
            änderung_verweildauer_chapter.sort_values(by = ("Wasserstein Distanz"), key = abs, ascending = True).head(5)
        )}

        Instabilste Kapitel:
        {self._to_typst_formatted(
            änderung_verweildauer_chapter.sort_values(by = ("Wasserstein Distanz"), key = abs, ascending = False).head(5)
        )}

        === Alter

        Wasserstein Distanz Änderung: {mean_wd_age:.3} Jahre; Standardabweichung: {std_wd_age:.3} Jahre

        Stabilste Gruppen:
        {self._to_typst_formatted(
            änderung_altersverteilung.sort_values(by = "Wasserstein Distanz", key = abs, ascending = True).head(5)
        )}

        Instabilste Gruppen:
        {self._to_typst_formatted(
            änderung_altersverteilung.sort_values(by = "Wasserstein Distanz", key = abs, ascending = False).head(5)
        )}

        Stabilste Kapitel:
        {self._to_typst_formatted(
            änderung_altersverteilung_chapter.sort_values(by = "Wasserstein Distanz", key = abs, ascending = True).head(5)
        )}

        Instabilste Kapitel:
        {self._to_typst_formatted(
            änderung_altersverteilung_chapter.sort_values(by = "Wasserstein Distanz", key = abs, ascending = False).head(5)
        )}

        === Änderung Nebendiagnosen

        Mittelwert Spearman: {mean_spear_nd:.3}; Standardabweichung: {std_spear_nd:.3}; Jaccard Score: {jaccard_nd:.3}

        Stabilste Gruppen:
        {self._to_typst_formatted(
            änderung_nd.sort_values(by = "Spearman r", key = abs, ascending = False).head(5)
        )}

        Instabilste Gruppen:
        {self._to_typst_formatted(
            änderung_nd.sort_values(by = "Spearman r", key = abs, ascending = True).head(5)
        )}

        Stabilste Kapitel:
        {self._to_typst_formatted(
            änderung_nd_chapter.sort_values(by = "Spearman r", key = abs, ascending = False).head(5)
        )}

        Instabilste Kapitel:
        {self._to_typst_formatted(
            änderung_nd_chapter.sort_values(by = "Spearman r", key = abs, ascending = True).head(5)
        )}


        """.replace("icd_group", "ICD-Gruppe").replace("icd_chapter", "ICD-Kapitel").replace("[],", "")

        with open(path, "w", encoding="utf-8") as f:
            f.write(auswertungsstring)

        self.analysis_df = original

        self.logger.info("Report erstellt.")
        self.logger.info("Stabilitätsanalyse abgeschlossen.")

    def _check_change_in_counts(self):
        """
        Berechnet die prozentuale Änderung der Fallzahlen zwischen dem Campus File (2010) und dem InEK-Datenbrowser (2024).
        """
        änderung_fallzahl =  (self.analysis_df[("Fallzahl", "", "datenbrowser_2024")] - self.analysis_df[("Fallzahl", "", "campus_file_2010")]) / self.analysis_df[("Fallzahl", "", "campus_file_2010")] * 100
        df = pd.DataFrame(änderung_fallzahl, columns = ["Änderung [Prozent]"])

        # auf Kapitelebene aggregieren
        df_chapter = (
            df
            .assign(
                icd_chapter=df.index.map(self.chapter_lookup),
                fallzahl=self.analysis_df[("Fallzahl", "", "campus_file_2010")]
            )
            .groupby("icd_chapter")
            .apply(lambda x: np.average(x["Änderung [Prozent]"], weights=x["fallzahl"]))
        )
        # Sicherstellen dass es eine Series ist
        if isinstance(df_chapter, pd.DataFrame):
            df_chapter = df_chapter.iloc[:, 0]

        df_chapter = df_chapter.to_frame(name="Änderung [Prozent]")
        return self._do_stat_analysis(änderung_fallzahl), df, df_chapter

    def _check_change_in_deaths(self):
        """
        Berechnet die Änderung der Mortalitätsrate in Prozentpunkten zwischen dem Campus File (2010) und dem InEK-Datenbrowser (2024).
        """
        mortalitätsrate_cf = self.analysis_df[("Todesfälle", "", "campus_file_2010")] / self.analysis_df[("Fallzahl", "", "campus_file_2010")] * 100
        mortalitätsrate_db = self.analysis_df[("Todesfälle", "", "datenbrowser_2024")] / self.analysis_df[("Fallzahl", "", "datenbrowser_2024")] * 100

        aenderung_todesfaelle = mortalitätsrate_db - mortalitätsrate_cf
        df = pd.DataFrame(aenderung_todesfaelle, columns = ["Änderung [Prozentpunkte]"])

        # auf Kapitelebene aggregieren
        df_chapter = (
            df
            .assign(
                icd_chapter=df.index.map(self.chapter_lookup),
                fallzahl=self.analysis_df[("Fallzahl", "", "campus_file_2010")]
            )
            .groupby("icd_chapter")
            .apply(lambda x: np.average(x["Änderung [Prozentpunkte]"], weights=x["fallzahl"]))
        )
        # Sicherstellen dass es eine Series ist
        if isinstance(df_chapter, pd.DataFrame):
            df_chapter = df_chapter.iloc[:, 0]

        df_chapter = df_chapter.to_frame(name="Änderung [Prozentpunkte]")
        return self._do_stat_analysis(aenderung_todesfaelle), df, df_chapter

    def _check_change_in_sex(self):
        """
        Berechnet die Änderung der Geschlechterverteilung in Prozentpunkten zwischen dem Campus File (2010) und dem InEK-Datenbrowser (2024).
        """
        # Excel-Export enthält Prozentwerte als String mit geschütztem Leerzeichen vor "%", z.B. "12,3\xa0%"
        # manche Zellen sind anders formatiert, dort dann 
        anteil_weiblich_db = pd.to_numeric(
            self.analysis_df[("Geschlecht", "weiblich", "datenbrowser_2024")]
            .astype(str).str.replace(",", ".").str.replace("\xa0%", "").str.strip()
        )
        anteil_weiblich_db.loc[anteil_weiblich_db <= 1] *= 100

        anteil_männlich_db = pd.to_numeric(
            self.analysis_df[("Geschlecht", "männlich", "datenbrowser_2024")]
            .astype(str).str.replace(",", ".").str.replace("\xa0%", "").str.strip()
        )
        anteil_männlich_db.loc[anteil_männlich_db <= 1] *= 100

        anteil_weiblich_cf = pd.to_numeric(self.analysis_df[("Geschlecht", "weiblich", "campus_file_2010")].astype(str).str.replace(",", ".").str.replace("\xa0%", "")) * 100
        anteil_männlich_cf = pd.to_numeric(self.analysis_df[("Geschlecht", "männlich", "campus_file_2010")].astype(str).str.replace(",", ".").str.replace("\xa0%", "")) * 100

        änderung_w = anteil_weiblich_db - anteil_weiblich_cf
        änderung_m = anteil_männlich_db - anteil_männlich_cf

        return self._do_stat_analysis(änderung_w), self._do_stat_analysis(änderung_m)
    
    def _check_change_in_los(self):
        """
        Berechnet die Änderung der Verweildauer in Tagen zwischen dem Campus File (2010) und dem InEK-Datenbrowser (2024).
        """
        # Quelle: Statistisches Bundesamt (2026). Diagnosedaten der Krankenhäuser nach Wohnsitz (ICD-10-3-Steller, ab 2000). GBE - Gesundheitsberichterstattung des Bundes. https://www.gbe-bund.de:443/gbe/isgbe.archiv?p_indnr=594&p_archiv_id=16298175&p_sprache=D&p_action=A (Abgerufen: 16. September 2026) 
        mean_vwd_longer_than21_d = np.average([np.mean([22, 42]), np.mean([43, 70]), np.mean([71, 182]), np.mean([183, 365])], weights=[881_856, 226_801, 89_561, 3_139])

        # Ordinale Verweildauerkategorien des Campus Files auf Klassenmitten in Tagen abgebildet
        # Kategorie 7 (>21 Tage): gewichtetes Mittel aus GBE-Bund Daten
        umrechnungs_dict_vwd= {
            1: .5,
            2: 2,
            3: 5.5,
            4: 9,
            5: 12.5,
            6: 18,
            7: np.round(mean_vwd_longer_than21_d, 2)
        }
        umrechnungs_werte = list(umrechnungs_dict_vwd.values())

        # Ordinale Verweildauer mit Klassenmitten verrechnen zu Tagen
        vals = []
        for i in range(len(self.analysis_df)):
            row_dict = self.analysis_df[("Verweildauer", "", "campus_file_2010")].iloc[i]
            weights_list = [row_dict.get(k, 0) for k in umrechnungs_dict_vwd.keys()]
            if sum(weights_list) == 0:
                vals.append(0)
            else:
                vals.append(np.average(umrechnungs_werte, weights=weights_list))

        data = pd.DataFrame(self.analysis_df[("Verweildauer", "", "datenbrowser_2024")])
        data[("Verweildauer", "", "campus_file_2010")] = vals
        data = data.dropna()

        col = ("Wasserstein Distanz", "", "")
        data[col] = data[("Verweildauer", "", "datenbrowser_2024")] - data[("Verweildauer", "", "campus_file_2010")]

        # auf Kapitelebene aggregieren
        df_chapter = (
            data
            .assign(
                icd_chapter=data.index.map(self.chapter_lookup),
                fallzahl=self.analysis_df[("Fallzahl", "", "campus_file_2010")]
            )
            .groupby("icd_chapter")
            .apply(lambda x: np.average(x["Wasserstein Distanz"], weights=x["fallzahl"]))
        )
        # Sicherstellen dass es eine Series ist
        if isinstance(df_chapter, pd.DataFrame):
            df_chapter = df_chapter.iloc[:, 0]

        df_chapter = df_chapter.to_frame(name="Wasserstein Distanz")

        return wasserstein_distance(data[("Verweildauer", "", "campus_file_2010")], data[("Verweildauer", "", "datenbrowser_2024")]) * np.sign(np.nanmean(data[("Verweildauer", "", "datenbrowser_2024")]) - np.nanmean(data[("Verweildauer", "", "campus_file_2010")])), data[["Wasserstein Distanz"]], df_chapter
    
    def _check_change_in_age(self):
        """
        Berechnet die Änderung der Altersstruktur in Jahren zwischen dem Campus File (2010) und dem InEK-Datenbrowser (2024).
        """
        alter_cf = self.analysis_df[("Alter", "", "campus_file_2010")].copy(deep = True)
        alter_db = self.analysis_df[("Alter", "", "datenbrowser_2024")].copy(deep = True)

        # Quelle: Statistisches Bundesamt (2026). Diagnosedaten der Krankenhäuser nach Wohnsitz (ICD-10-3-Steller, ab 2000). GBE - Gesundheitsberichterstattung des Bundes. https://www.gbe-bund.de:443/gbe/isgbe.archiv?p_indnr=594&p_archiv_id=16298175&p_sprache=D&p_action=A (Abgerufen: 16. September 2026) 
        average_age_older_than_80 = np.average([np.mean([80, 84]), np.mean([85, 89]), np.mean([90, 94]), np.mean([95, 99])], weights = [1_417_961, 918_644, 264_339, 78_019])

        # Altersklassen von Campus File und Datenbrowser harmonisiert auf gemeinsames Schema
        # Klassenmitten für Wasserstein-Distanz-Berechnung, >80 Jahre: gewichtetes Mittel
        klassenmitten = np.array([0.5, 5.5, 20.0, 35.0, 45.0, 55.0, 70.0, average_age_older_than_80])

        wds = []

        for i in range(len(self.analysis_df)):

            if alter_db.iloc[i] is None or alter_cf.iloc[i] is None:
                wds.append(np.nan)
                continue

            # Fallzahlen extrahieren
            fallzahl = self.analysis_df.iloc[i][("Fallzahl", "", "datenbrowser_2024")]
            alter_db_new = {
                "<1 Jahr": (pd.to_numeric(str(alter_db.iloc[i].get("< 28 Tage", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("28 Tage - 1. Jahr", 0)).replace(",", ".").replace("\xa0%", "")))*fallzahl,
                ">= 1 und < 10 Jahre": (pd.to_numeric(str(alter_db.iloc[i].get("1-2 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("3-5 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("6-9 Jahre", 0)).replace(",", ".").replace("\xa0%", "")))*fallzahl,
                ">= 10 und < 30 Jahre": (pd.to_numeric(str(alter_db.iloc[i].get("10-15 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("16-17 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("18-29 Jahre", 0)).replace(",", ".").replace("\xa0%", "")))*fallzahl,
                ">= 30 und < 40 Jahre": pd.to_numeric(str(alter_db.iloc[i].get("30-39 Jahre", 0)).replace(",", ".").replace("\xa0%", ""))*fallzahl,
                ">= 40 und < 50 Jahre": pd.to_numeric(str(alter_db.iloc[i].get("40-49 Jahre", 0)).replace(",", ".").replace("\xa0%", ""))*fallzahl,
                ">= 50 und < 60 Jahre": (pd.to_numeric(str(alter_db.iloc[i].get("50-54 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("55-59 Jahre", 0)).replace(",", ".").replace("\xa0%", "")))*fallzahl,
                ">= 60 und < 80 Jahre": (pd.to_numeric(str(alter_db.iloc[i].get("60-64 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("65-74 Jahre", 0)).replace(",", ".").replace("\xa0%", "")) + pd.to_numeric(str(alter_db.iloc[i].get("75-79 Jahre", 0)).replace(",", ".").replace("\xa0%", "")))*fallzahl,
                ">= 80 Jahre": pd.to_numeric(str(alter_db.iloc[i].get("80+ Jahre", 0)).replace(",", ".").replace("\xa0%", ""))*fallzahl
            }
            alter_cf_new = {
                "<1 Jahr": alter_cf.iloc[i].get(1, 0),
                ">= 1 und < 10 Jahre": alter_cf.iloc[i].get(2, 0),
                ">= 10 und < 30 Jahre": alter_cf.iloc[i].get(3, 0) + alter_cf.iloc[i].get(4, 0),
                ">= 30 und < 40 Jahre": alter_cf.iloc[i].get(5, 0),
                ">= 40 und < 50 Jahre": alter_cf.iloc[i].get(6, 0),
                ">= 50 und < 60 Jahre": alter_cf.iloc[i].get(7, 0),
                ">= 60 und < 80 Jahre": alter_cf.iloc[i].get(8, 0) + alter_cf.iloc[i].get(9, 0),
                ">= 80 Jahre": alter_cf.iloc[i].get(10, 0),
            }

            df_alter_db_new = pd.DataFrame(alter_db_new, index=[0]).T
            df_alter_db_new.columns = ["datenbrowser_2024"]

            df_alter_cf_new = pd.DataFrame(alter_cf_new, index=[0]).T
            df_alter_cf_new.columns = ["campus_file_2010"]

            df_combined = df_alter_db_new.join(df_alter_cf_new)

            # Mittleres Alter berechnen mittels Klassenmitte
            mean_cf = np.average(klassenmitten, weights = df_combined["campus_file_2010"])
            mean_db = np.average(klassenmitten, weights = df_combined["datenbrowser_2024"])

            wds.append(wasserstein_distance(
                u_values=klassenmitten, 
                v_values=klassenmitten, 
                u_weights=df_combined["campus_file_2010"], 
                v_weights=df_combined["datenbrowser_2024"]
            ) * np.sign(mean_db - mean_cf))

        df = pd.DataFrame(wds, index=self.analysis_df.index, columns=["Wasserstein Distanz"])

        # auf Kapitelebene aggregieren
        df_chapter = (
            df
            .assign(
                icd_chapter=df.index.map(self.chapter_lookup),
                fallzahl=self.analysis_df[("Fallzahl", "", "campus_file_2010")]
            )
            .groupby("icd_chapter")
            .apply(lambda x: np.average(x["Wasserstein Distanz"], weights=x["fallzahl"]))
        )
        # Sicherstellen dass es eine Series ist
        if isinstance(df_chapter, pd.DataFrame):
            df_chapter = df_chapter.iloc[:, 0]

        df_chapter = df_chapter.to_frame(name="Wasserstein Distanz")
        return self._do_stat_analysis(df["Wasserstein Distanz"]), df, df_chapter
    
    def _calc_spear(self, column):
        """
        Berechnet Spearman-Rangkorrelation und Jaccard-Distanz zwischen Campus File (2010) und Datenbrowser (2024) für eine gegebene Spalte (Nebendiagnosen oder Prozeduren).

        Args:
            column: Spaltenname im analysis_df, z.B. 'Nebendiagnosen'
        Returns:
            Tuple aus (Spearman-r-Liste, Jaccard-Liste, DataFrame auf Gruppenebene, DataFrame auf Kapitelebene)
        """
        res_results = []
        jaccards = []
        valid_icd_groups = []

        # über alle ICD-Gruppen iterieren
        for icd_group in self.analysis_df.index:
            if self.analysis_df[(column, "", "datenbrowser_2024")][icd_group] is None:
                continue
            # Daten extrahieren
            dict_nd_db = dict(zip(self.analysis_df[(column, "", "datenbrowser_2024")][icd_group].keys(), np.array(list(self.analysis_df[(column, "", "datenbrowser_2024")][icd_group].values()))))

            dict_nd_cf = dict(zip(self.analysis_df[(column, "", "campus_file_2010")][icd_group].keys(), np.array(list(self.analysis_df[(column, "", "campus_file_2010")][icd_group].values()))))

            df_nd_db = pd.DataFrame(dict_nd_db, index=[0]).T
            df_nd_db.columns = ["datenbrowser_2024"]

            df_nd_cf = pd.DataFrame(dict_nd_cf, index=[0]).T
            df_nd_cf.columns = ["campus_file_2010"]

            # Daten zusammenfügen
            df_combined = df_nd_db.join(df_nd_cf, how = 'outer').fillna(0)
            df_combined_jaccard = df_nd_db.join(df_nd_cf, how = 'outer').fillna(0)
            df_combined_jaccard = df_combined_jaccard.astype(bool)

            # "---" steht für Nebendiagnosen, die nicht vorhanden sind, entfernen, da sie die Berechnung der Jaccard-Distanz verzerren würden
            if "---" in df_combined_jaccard.index:
                df_combined_jaccard = df_combined_jaccard.drop("---")
            if "---" in df_combined.index:
                df_combined = df_combined.drop("---")

            # Jaccard Score berechnen
            set_2024 = set(df_combined_jaccard[df_combined_jaccard['datenbrowser_2024'] == True].index)
            set_2010 = set(df_combined_jaccard[df_combined_jaccard['campus_file_2010'] == True].index)
            jaccard = 1 - (len(set_2024.intersection(set_2010)) / len(set_2024.union(set_2010)))

            # Spearmean r berechnen
            res = stats.spearmanr(df_combined["campus_file_2010"], df_combined["datenbrowser_2024"])[0]

            res_results.append(res)
            jaccards.append(jaccard)
            valid_icd_groups.append(icd_group)
        df = pd.DataFrame(res_results, index=valid_icd_groups, columns = ["Spearman r"])

        # auf Kapitelebene aggregieren
        df_chapter = (
            df
            .assign(
                icd_chapter=df.index.map(self.chapter_lookup),
                fallzahl=self.analysis_df[("Fallzahl", "", "campus_file_2010")]
            )
            .groupby("icd_chapter")
            .apply(lambda x: np.average(x["Spearman r"], weights=x["fallzahl"]))
        )
        # Sicherstellen dass es eine Series ist
        if isinstance(df_chapter, pd.DataFrame):
            df_chapter = df_chapter.iloc[:, 0]

        df_chapter = df_chapter.to_frame(name="Spearman r")

        return res_results, jaccards, df, df_chapter
    
    def _check_change_in_nd(self):
        """
        Wrapper Funktion zum Berechnen der Änderung der Kodierung der Nebendiagnosen
        """
        ress, jaccard, df, df_chapter = self._calc_spear(column = "Nebendiagnosen")

        jaccard_series = pd.Series(jaccard, index=df.index)
        weights = self.analysis_df[("Fallzahl", "", "campus_file_2010")].loc[df.index]

        return self._do_stat_analysis(df["Spearman r"]), np.average(jaccard_series, weights=weights), df, df_chapter
    
    
    def _do_stat_analysis(self, vals):
        """
        Berechnet den Mittelwert und Interquartilsabstand einer Verteilung.
        """
        vals = pd.Series(vals).dropna()

        weights = self.analysis_df[("Fallzahl", "", "campus_file_2010")].loc[vals.index]
        weighted_mean = np.average(vals, weights=weights)
        variance = np.average((vals - weighted_mean)**2, weights=weights)
        return weighted_mean, np.sqrt(variance)