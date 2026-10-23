"""
Vorverarbeitung des Campus Files DRG 2010: Laden, Bereinigen, Plausibilitätsprüfung und Speicherung als Multi-Hot-kodierte Feather-Dateien für das Modelltraining.
"""

import pandas as pd
import json
from collections import Counter
import os

from drg_risk import DATA_DIR
from drg_risk.utils.logger import Logger

class DataPreprocessor:
    """
    Klasse zur Aufbereitung der Daten des Campus-Files.
    Daten werden geladen, aufbereitet, auf Plausibilität geprüft und als feather Datei gespeichert.
    """
    def __init__(self, data_path, icd_sys_path, ops_sys_path, columns_metadata_path, subgroups_path, drg_path, min_count_per_column = 20):
        """
        Args:
            data_path: Pfad zum Campus File (CSV)
            icd_sys_path: Pfad zur ICD-Systematik (JSON)
            ops_sys_path: Pfad zur OPS-Systematik (JSON)
            columns_metadata_path: Pfad zur Spaltenbeschreibung (JSON)
            subgroups_path: Pfad zu ICD-Subgruppen (JSON)
            drg_path: Pfad zum Fallpauschalenkatalog (Excel)
            min_count_per_column: Mindesfallzahl pro ICD-Code, Standard 20
        """
        self.data_path =                data_path
        self.icd_sys_path =             icd_sys_path
        self.ops_sys_path =             ops_sys_path
        self.columns_metadata_path =    columns_metadata_path
        self.subgroups_path =           subgroups_path
        self.drg_path =                 drg_path
        self.logger =                   Logger(name = "dataprocessing")
        self.MIN_COUNT_PER_COLUMN =     min_count_per_column

        self.logger.info("Initialisierung des Processors abgeschlossen.")

    def _load_campus_file(self):
        """
        Lädt das rohe Campus File und gibt es als DataFrame zurück.
        """
        self.logger.info("Lade Original Campus File.")

        with open(self.data_path, "r", encoding="utf-8") as f:
            cf = pd.read_csv(f, delimiter = ";")

        self.logger.info("Originales Campus File geladen.")
        return cf
        
    def _load_json_data(self, path):
        """
        Funktion zum Laden von JSON Dateien.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    def _load_cleaned_icd_groups(self):
        """
        Bereinigt die ICD-Gruppen von den Subgruppen, da manche ICD-Gruppen unterschiedlich aufgeschlüsselt wurden zwischen dem Campus-File und der ICD-Systematik des bifg.
        """
        self.logger.info("Lade bereinigte ICD-Gruppen.")

        # Lädt die ICD-Systematik und die Subgruppen
        self.icd_systematik = self._load_json_data(self.icd_sys_path)
        subgroups_dict = self._load_json_data(self.subgroups_path)

        # Gruppen als Liste konvertieren
        icd_groups = list(set([group["number"] for chapter in self.icd_systematik.values() for group in chapter["groups"].values()]))

        # über Subgruppen iterieren
        for group, subgroups in subgroups_dict.items():
            for subgroup in subgroups:
                # Subgruppen aus der Liste der Gruppen entfernen
                icd_groups.remove(subgroup)
            icd_groups.append(group)

        self.logger.info("Bereinigte ICD-Gruppen geladen.")
        return icd_groups

    def _load_cleaned_ops_groups(self):
        """
        Lädt OPS Systematik und konvertiert diese in eine Liste.
        """
        self.logger.info("Lade bereinigte OPS-Gruppen.")

        # OPS Systematik laden
        ops_systematik = self._load_json_data(self.ops_sys_path)
        # OPS Gruppen in Liste konvertieren
        ops_groups = list(set([f"{group["number_start"]}-{group["number_end"]}" for chapter in ops_systematik.values() for group in chapter["groups"].values()]))

        self.logger.info("Bereinigte OPS-Gruppen geladen.")
        return ops_groups
    
    def _load_cleaned_drg_codes(self):
        """
        Lädt die DRG-Codes, bereinigt sie und gibt die Codes in einer Liste zurück.
        """
        self.logger.info("Lade bereinigte DRG-Codes.")
        sheets = ["Hauptabteilungen", "Belegabteilungen", "Anlage 3a"]

        drg_codes = []
        # Iteriere über mehrere Blätter der Excel Datei.
        for sheet in sheets:
            excel_data = pd.read_excel(self.drg_path, sheet_name=sheet, skiprows=7, names = ["DRG"], usecols="A")
            # DRG-Codes der Liste anhängen
            drg_codes.extend(excel_data["DRG"].unique())

        # Aufbereitung der DRG-Codes
        drg_codes = [code for code in drg_codes if isinstance(code, str) and "MDC" not in code]
        drg_codes = [code.split(" ")[0] for code in drg_codes]
        drg_codes = [code for code in drg_codes if len(code) == 4]

        self.logger.info("Bereinigte DRG-Codes geladen.")
        return drg_codes
    
    def _load_columns_metadata(self):
        """
        Lädt Spalten-Metadaten und ergänzt sie um gültige Ausprägungen für ICD-, OPS- und DRG-Spalten aus den bereits geladenen Systematiken.

        Returns:
            Dict {Spaltenname: {dtype, mandatory, true_values}}
        """
        self.logger.info("Lade Metadaten der Spalten.")

        # Lade Metadaten
        columns_metadata = self._load_json_data(self.columns_metadata_path)

        columns_metadata_alphanumeric = {}
        # Iteriere über die Spalten
        for column, items in columns_metadata.items():
            columns_metadata_alphanumeric[column] = {}
            # ICD-Gruppen, OPS-Gruppen und DRG-Codes als richtige Ausprägungen für diese Variablen.
            if column.startswith("icd"):
                items["true_values"] = {group: group for group in self.icd_groups.copy()}
            elif column.startswith("ops"):
                items["true_values"] = {group: group for group in self.ops_groups.copy()}
            elif column.startswith("drg"):
                items["true_values"] = {code: code for code in self.drg_codes.copy()}

            # Iteriere über die Anforderungen an Variablen
            for name, item in items.items():
                columns_metadata_alphanumeric[column][name] = {}
                # Metadaten speichern in Dictionary
                if name == "true_values":
                    # Wenn keine richtigen Werte vorhanden, dann None
                    if items["true_values"] == None:
                        columns_metadata_alphanumeric[column]["true_values"] = None
                        continue
                    for key in item.keys():
                        # Numerische Keys (z.B. ICD-Kapitelnummern) in int konvertieren, alphanumerische Codes (ICD/OPS/DRG) bleiben String
                        key_alpha_num = int(key) if key.isnumeric() else key
                        columns_metadata_alphanumeric[column]["true_values"][key_alpha_num] = items["true_values"][key]
                else:
                    columns_metadata_alphanumeric[column][name] = items[name]
        self.logger.info("Metadaten geladen.")
        # Metadaten zurückgeben
        return columns_metadata_alphanumeric
    
    def _enforce_dtype(self):
        """
        Konvertiert den Datentyp zum richtigen Datentyp aus den Metadaten.
        """
        self.logger.info("Erzwinge Datentyp.")
        for col, values in self.columns_metadata.items():
            self.df[col] = self.df[col].astype(values["dtype"])
        self.logger.info("Datentyp erzwungen.")

    def _drop_nan(self):
        """
        Entfernt Krankenhausfälle, bei denen Wert von Variablen, die notwendig sind, nicht vorhanden sind.
        """
        self.logger.info("Entferne Krankenhausfälle mit Fehlerwerten.")

        # Variablen, die notwendig sind; Nebendiagnose 10 ist beispielsweise nicht notwendig
        mandatory_columns = [col for col, value in self.columns_metadata.items() if value.get("mandatory", False)]
        self.df = self.df.dropna(subset=mandatory_columns)

        self.logger.info("Fehlerwerte bereinigt.")

    def _monitor_true_values(self):
        """
        Überprüft, ob Variablen nur die richtigen Ausprägungen aus den Metadaten haben.
        Wenn nicht wird eine Fehlermeldung ausgegeben.
        """
        self.logger.info("Überprüfe mögliche Feldausprägungen.")

        df_temporary = self.df.copy()

        wrong_icd = set()
        wrong_ops = set()
        wrong_drg = set()

        # Über Metadaten iterieren
        for column, values in self.columns_metadata.items():
            true_values = values["true_values"]
            if true_values == None:
                # Wenn numerische Variable < 0 ist, ist sie falsch.
                falsche_daten = df_temporary[df_temporary[column]<0]
            else:
                true_values = true_values.keys()
                # Daten sind richtig, wenn sie in den Metadaten definiert sind, oder keine Daten vorhanden sind bei eienr nicht notwendigen Variable.
                if values["mandatory"]:
                    falsche_daten = df_temporary[~(df_temporary[column].isin(true_values))]
                else:
                    falsche_daten = df_temporary[~((df_temporary[column].isin(true_values) | df_temporary[column].isna()))]

            if falsche_daten.shape[0] == 0:
                continue
            else:
                # Zusammenfassen von falschen Daten wenn ICD-Code, OPS-Code oder DRG-Code
                if column.startswith("icd"):
                    wrong_icd.update(list(falsche_daten[column].unique()))
                elif column.startswith("ops"):
                    wrong_ops.update(list(falsche_daten[column].unique()))
                elif column.startswith("drg"):
                    wrong_drg.update(list(falsche_daten[column].unique()))
                else:
                    print(f"In der Spalte {column} gibt es falsche Werte: {list(falsche_daten[column].unique())}")

        # Ausgabe einer Fehlermeldung, wenn ICD-Code, OPS-Code oder DRG-Code falsch sind.
        if len(wrong_icd) > 0:
            print(f"Folgende ICD Codes weichen von der ICD-Systematik des BfArM ab: {sorted(wrong_icd)}")
        else:
            print("Kein ICD Code weicht von der ICD-Systematik des BfArM ab.")
        
        if len(wrong_ops) > 0:
            print(f"Folgende OPS Codes weichen von der OPS-Systematik des BfArM ab: {sorted(wrong_ops)}")
        else:
            print("Kein OPS Code weicht von der OPS-Systematik des BfArM ab.")
        
        if len(wrong_drg) > 0:
            print(f"Folgende DRG Codes weichen von der DRG-Systematik des InEK ab: {sorted(wrong_drg)}")
        else:
            print("Kein DRG Code weicht von der DRG-Systematik des InEK ab.")

        self.logger.info("Überprüfung der Feldausprägung abgeschlossen.")

    def _monitor_error_drg(self):
        """
        Überprüft DRG-Codes nach Fehler-DRG. Diese werden dokumentiert und anschließend entfernt.
        """
        self.logger.info("Überprüfe Fehler-DRG.")

        df_temporary = self.df.copy()
        error_drg = [
            "960Z",
            "961Z",
            "962Z",
            "963Z"
        ]
        # Prüfen ob DRG-Code ein Fehler DRG-Code ist.
        error_cases = list(df_temporary[df_temporary["drgh"].isin(error_drg)]["drgh"])
        if len(error_cases) > 0:
            counter = Counter(sorted(error_cases))
            # Wenn Fehler DRG-Codes gefunden wurden, Fehlermeldung ausgeben.
            for drg, count in counter.items():
                print(f"DRG Code {drg} wurde {count}x gefunden.")
            print(f"Insgesamt gibt es {len(error_cases)} Fälle mit Fehler DRG.")

        # Krankenhausfälle mit Fehler DRG-Codes entfernen.
        self.df = self.df[~self.df["drgh"].isin(error_drg)]

        self.logger.info("Überprüfung nach Fehler-DRG abgeschlossen.")

    def _save_ungrouped_df(self):
        """
        Speichert das Campus File als Feather Datei.
        """
        self.logger.info("Speichere ungruppierte Daten zu Feather.")

        self.df.to_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "ungrouped_df.feather"))

        self.logger.info("Speichern abgeschlossen.")

    def _dummify_groups_chapters(self, df):
        """
        Multi-Hot-kodiert ICD-Hauptdiagnose und alle Nebendiagnosen auf Gruppenebene.
        Nebendiagnosen werden akkumuliert: ein Fall erhält 1 für eine Gruppe wenn mindestens eine seiner Diagnosen in diese Gruppe fällt.

        Args:
            df: DataFrame mit ICD-Spalten
        Returns:
            DataFrame mit bool Multi-Hot-Spalten pro ICD-Gruppe
        """
        self.logger.info("Erstelle dummy Spalten für ICD-Gruppen.")
        # Variablen, die nicht Multi-Hot-kodiert werden sollen.
        base_df = df.copy()
        base_df = base_df[[col for col in df.columns if "icd" not in col]]
        base_df = base_df.set_index("fall_nr")

        # Variablen, die Multi-Hot-kodiert werden sollen.
        icd_df = df.copy()
        icd_df = icd_df[[col for col in self.columns_df if "icd" in col or "fall_nr" in col]]
        icd_df = icd_df.set_index("fall_nr")

        # Hauptdiagnosen Multi-Hot-kodieren.
        small_df_hd = icd_df[["icd_hd3"]]
        dummy_df_hd = pd.get_dummies(small_df_hd, prefix="icd_group_is", prefix_sep="_", columns = ["icd_hd3"])

        merged = dummy_df_hd.copy()

        # Iteriert über die Nebendiagnosen
        for sdx in self.ndx_columns:
            small_df_nd = icd_df[[sdx]]
            # Nebendiagnosen Multi-Hot-kodieren.
            dummy_df_nd = pd.get_dummies(small_df_nd, prefix="icd_group_is", prefix_sep="_", columns = [sdx])

            # Multi-Hot-kodierte Nebendiagnosen an finales DataFrame anfügen
            # Boolesche ODER-Verknüpfung via Addition: Fall bekommt 1, sobald mind. eine seiner Diagnosen in die Gruppe fällt
            merged = merged.add(dummy_df_nd, fill_value=False).astype(bool)

        # Variablen, die nicht Multi-Hot-kodiert werden sollen, mit Variablen, die Multi-Hot-kodiert werden sollen, zusammenfügen
        merged = pd.merge(base_df, merged, left_index=True, right_index=True)

        self.logger.info("Erstellung der dummy Spalten beendet.")
        return merged
    
    def _add_chapter_columns(self, df):
        """
        Fügt die Kapitel Ebene mit Multi-Hot-Kodierung hinzu.
        Diese stellt die zweite Granularitätsstufe dar.
        """
        self.logger.info("Erstelle dummy Spalten für ICD-Kapitel.")
        # ICD-Kapitel aus der ICD-Systematik
        icd_chapters = [chapter for chapter in self.icd_systematik.values()]

        self.chapter_series  = {}
        # über Kapitel iterieren
        for icd_chapter in icd_chapters:
            group_numbers = [group["number"] for group in icd_chapter["groups"].values()]
            columns = [f"icd_group_is_{g}" for g in group_numbers if f"icd_group_is_{g}" in df.columns]
            if columns:
                self.chapter_series[f"icd_chapter_is_{icd_chapter['number']}"] = df[columns].any(axis=1)

        df = pd.concat([df, pd.DataFrame(self.chapter_series, index=df.index)], axis=1)
        self.logger.info("Erstellung der dummy Spalten beendet.")
        return df
    
    def _get_cm_score(self, icd_structure, score):
        """
        Gibt Charlsonlike- oder Elixhauserlike-Score einer ICD-Struktur (Gruppe oder Kapitel) zurück.
        Lookup-Tabelle wird aus comorbidity_lookup.json geladen (ICD-Prävalenzen aus InEK-Datenbrowser 2024).

        Args:
            icd_structure: "Gruppe {icd_gruppen_code}" oder "Kapitel {icd_kapitel_nr}"
            score: 'charlson' oder 'elixhauser'
        Returns:
            Score als Zahl, 0 wenn Gruppe/Kapitel nicht im Lookup
        """
        scores = self.com_lookup.get(icd_structure, {})
        return scores.get(score, 0)

    def _build_exclusion_sets(self):
        """
        Erstellt die Listen von zu exkludierenden Gruppen und Spalten.
        Betroffene ICD-Kapitel die für die Risikoadjustierung nicht geeignet sind:
            1, 15, 20, 21, 22
        Ermöglicht direkten Vergleich mit Ergebnissen von Stausberg und Hagn.
        """
        # Auszuschließende Kapitel
        chapters_to_exclude = [
            1, 15, 20, 21, 22
        ]
        
        # ICD-Systematik laden
        self.icd_systematik = self._load_json_data(self.icd_sys_path)

        # zu exkludierende Gruppen und Kapitel
        groups_to_exclude = []
        for chapter in chapters_to_exclude:
            for key, item_c in self.icd_systematik.items():
                if item_c["number"] == chapter:
                    for group, item_g in item_c["groups"].items():
                        groups_to_exclude.append(item_g["number"])
        groups_to_exclude.append("O80-O84")
        groups_to_exclude.append("O95-O99")
        groups_to_exclude.append("R95-R99")
        cols_to_exclude = set(
            [f"icd_group_is_{g}"   for g in groups_to_exclude] +
            [f"icd_chapter_is_{c}" for c in chapters_to_exclude]
        )

        return groups_to_exclude, cols_to_exclude

    def _get_low_prevalence_cols(self):
        """
        Erkennt ICD-Gruppen mit geringer Prävalenz (weniger als 15 Fälle)
        """

        # Anzahl erhalten
        counts_pdx     = self.df["icd_hd3"].value_counts()
        counts_pdx_sdx = pd.concat(
            [self.df[col] for col in self.df.columns if col.startswith("icd_hd") or col.startswith("icd_nd")]
        ).value_counts()

        # Gruppen basierend auf Anzahl exkludieren
        low_prev_pdx_num = [g for g, c in counts_pdx.items()     if c < self.MIN_COUNT_PER_COLUMN]
        low_prev_pdx_sdx_num = [g for g, c in counts_pdx_sdx.items() if c < self.MIN_COUNT_PER_COLUMN]

        low_prev_pdx_col = [f"icd_group_is_{g}" for g in low_prev_pdx_num]
        low_prev_pdx_sdx_col = [f"icd_group_is_{g}" for g in low_prev_pdx_sdx_num]

        # zu exkludierende Gruppen hinzufügen
        self.groups_to_exclude_pdx     = set(self.groups_to_exclude) | set(low_prev_pdx_num)
        self.groups_to_exclude_pdx_sdx = set(self.groups_to_exclude) | set(low_prev_pdx_sdx_num)
        self.cols_to_exclude_pdx       = set(self.cols_to_exclude)   | set(low_prev_pdx_col)
        self.cols_to_exclude_pdx_sdx   = set(self.cols_to_exclude)   | set(low_prev_pdx_sdx_col)

    def _get_unbalanced_mortality_cols(self):
        """
        Erkennt ICD-Gruppen mit unausgeglichener Mortalität.
        Unausgeglichene Mortalität bedeutet weniger als 15 Überlebende oder weniger als 15 Sterbefälle.
        """
        self.logger.info("Entferne ICD-Gruppen und ICD-Kapitel mit unausgeglichener Mortalität.")
        deaths = self.df[self.df["entl_grd"] == 5].copy()
        survivors = self.df[self.df["entl_grd"] != 5].copy()

        icd_cols = [col for col in self.df.columns if col.startswith("icd_hd") or col.startswith("icd_nd")]

        counts_dead_pdx_sdx     = pd.concat([deaths[col]    for col in icd_cols]).value_counts()
        counts_survivor_pdx_sdx = pd.concat([survivors[col] for col in icd_cols]).value_counts()
        counts_dead_pdx         = deaths["icd_hd3"].value_counts()
        counts_survivor_pdx     = survivors["icd_hd3"].value_counts()

        unbalanced_pdx = set(
            [f"icd_group_is_{g}" for g, c in counts_dead_pdx.items()     if c < self.MIN_COUNT_PER_COLUMN] +
            [f"icd_group_is_{g}" for g, c in counts_survivor_pdx.items() if c < self.MIN_COUNT_PER_COLUMN]
        )
        unbalanced_pdx_sdx = set(
            [f"icd_group_is_{g}" for g, c in counts_dead_pdx_sdx.items()     if c < self.MIN_COUNT_PER_COLUMN] +
            [f"icd_group_is_{g}" for g, c in counts_survivor_pdx_sdx.items() if c < self.MIN_COUNT_PER_COLUMN]
        )

        self.cols_to_exclude_mortality_pdx     = self.cols_to_exclude_pdx     | unbalanced_pdx
        self.cols_to_exclude_mortality_pdx_sdx = self.cols_to_exclude_pdx_sdx | unbalanced_pdx_sdx
        self.groups_to_exclude_mortality_pdx     = self.groups_to_exclude_pdx     | {c.split("icd_group_is_")[-1] for c in unbalanced_pdx}
        self.groups_to_exclude_mortality_pdx_sdx = self.groups_to_exclude_pdx_sdx | {c.split("icd_group_is_")[-1] for c in unbalanced_pdx_sdx}
        self.logger.info("ICD-Gruppen und ICD-Kapitel mit unausgeglichener Mortalität entfernt.")

    def _compute_anzahl_nebendiagnosen(self, df, groups_to_exclude):
        """
        Berechnet die Anzahl der Nebendiagnosen eines Falles.
        Berücksichtigt dabei nur Diagnosen, die nicht ausgeschlossen wurden.

        Args:
            df: Datensatz
        """
        ndx_cols = [col for col in df.columns if col.startswith("icd_nd")]
        # Nur Diagnosen inkludieren, die nicht ausgeschlossen wurden.
        df["anzahl_nebendiagnosen"] = (
            df[ndx_cols]
            .apply(lambda col: col.map(lambda x: x not in groups_to_exclude if pd.notna(x) else False))
            .sum(axis=1)
        )
        return df

    def _compute_comorbidity_scores(self, df, cols_to_exclude):
        """
        Berechnet die Komorbiditätsscores.
        Berücksichtigt dabei nur Diagnosen, die nicht ausgeschlossen wurden.

        Args:
            df: Datensatz
        """

        df = df.copy()
        group_cols   = [col for col in df.columns if "icd_group_is_"   in col and col not in cols_to_exclude]
        chapter_cols = [col for col in df.columns if "icd_chapter_is_" in col and col not in cols_to_exclude]

        group_names   = [col.split("icd_group_is_")[-1]   for col in group_cols]
        chapter_names = [col.split("icd_chapter_is_")[-1] for col in chapter_cols]

        df["charlson_group"]     = df[group_cols].dot([self._get_cm_score(f"Gruppe {g}",   "charlson")   for g in group_names])
        df["elixhauser_group"]   = df[group_cols].dot([self._get_cm_score(f"Gruppe {g}",   "elixhauser") for g in group_names])
        df["charlson_chapter"]   = df[chapter_cols].dot([self._get_cm_score(f"Kapitel {c}", "charlson")   for c in chapter_names])
        df["elixhauser_chapter"] = df[chapter_cols].dot([self._get_cm_score(f"Kapitel {c}", "elixhauser") for c in chapter_names])

        return df

    def _process_pdx(self, process_mortality=False):
        """
        Wrapper Funktion zur Verarbeitung unter Verwendung der Hauptdiagnose.
        """
        self.logger.info("Verarbeitung unter Verwendung der Hauptdiagnose.")
        self.res_pdx = self.df.copy(deep=True)
        if process_mortality:
            self.res_pdx = self._compute_anzahl_nebendiagnosen(self.res_pdx, self.groups_to_exclude_mortality_pdx)
        else:
            self.res_pdx = self._compute_anzahl_nebendiagnosen(self.res_pdx, self.groups_to_exclude_pdx)
        self.res_pdx = pd.get_dummies(self.res_pdx, columns=["icd_hd3"], prefix="icd_group_is", prefix_sep="_")
        self.res_pdx = self.res_pdx.drop(columns=self.ndx_columns)
        self.res_pdx = self._add_chapter_columns(self.res_pdx)
        if process_mortality:
            self.res_pdx = self._compute_comorbidity_scores(self.res_pdx, self.cols_to_exclude_mortality_pdx)
            self.res_pdx = self.res_pdx.drop(columns=[c for c in self.cols_to_exclude_mortality_pdx if c in self.res_pdx.columns])
        else:
            self.res_pdx = self._compute_comorbidity_scores(self.res_pdx, self.cols_to_exclude_pdx)
            self.res_pdx = self.res_pdx.drop(columns=[c for c in self.cols_to_exclude_pdx if c in self.res_pdx.columns])

        self.res_pdx = self.res_pdx.copy()

        self.logger.info("Verarbeitung unter Verwendung der Hauptdiagnose abgeschlossen.")

    def _process_pdx_sdx(self, process_mortality=False):
        """
        Wrapper Funktion zur Verarbeitung unter Verwendung der Hauptdiagnose.
        """
        self.logger.info("Verarbeitung unter Verwendung der Hauptdiagnose und Nebendiagnosen.")
        self.res_pdx_sdx = self.df.copy(deep=True)
        if process_mortality:
            self.res_pdx_sdx = self._compute_anzahl_nebendiagnosen(self.res_pdx_sdx, self.groups_to_exclude_mortality_pdx_sdx)
        else:
            self.res_pdx_sdx = self._compute_anzahl_nebendiagnosen(self.res_pdx_sdx, self.groups_to_exclude_pdx_sdx)
        self.res_pdx_sdx = self._dummify_groups_chapters(self.res_pdx_sdx)
        self.res_pdx_sdx = self.res_pdx_sdx.reset_index()
        self.res_pdx_sdx = self.res_pdx_sdx.drop(columns=self.ndx_columns, errors="ignore")
        self.res_pdx_sdx = self._add_chapter_columns(self.res_pdx_sdx)
        if process_mortality:
            self.res_pdx_sdx = self._compute_comorbidity_scores(self.res_pdx_sdx, self.cols_to_exclude_mortality_pdx_sdx )
            self.res_pdx_sdx = self.res_pdx_sdx.drop(columns=[c for c in self.cols_to_exclude_mortality_pdx_sdx if c in self.res_pdx_sdx.columns])
        else:
            self.res_pdx_sdx = self._compute_comorbidity_scores(self.res_pdx_sdx, self.cols_to_exclude_pdx_sdx)
            self.res_pdx_sdx = self.res_pdx_sdx.drop(columns=[c for c in self.cols_to_exclude_pdx_sdx if c in self.res_pdx_sdx.columns])

        self.res_pdx_sdx = self.res_pdx_sdx.copy()
        self.logger.info("Verarbeitung unter Verwendung der Hauptdiagnose und Nebendiagnosen abgeschlossen.")

    def _save_grouped_df_by_icd(self):
        """
        Fügt die Anzahl der Nebendiagnosen als Kovarianze hinzu.
        Fügt die Proxy-Komorbidität-Scores als Eingabevariablen hinzu.
        Erzeugt eine Multi-Hot-Kodierung der Diagnosen.
        Speicher die Daten als schnell lesbare Feather Dateien.
        """
        self.logger.info("Speichere Daten zu Feather.")

        self.columns_df = self.df.columns
        columns_to_dummify = [col for col in self.columns_df if "icd" in col]
        self.ndx_columns = [col for col in columns_to_dummify if "nd" in col]

        # Lade Komorbiditäts Verzeichnis
        with open(os.path.join(DATA_DIR, "comorbidity_lookup.json"), "r") as f:
            self.com_lookup = json.load(f)

        self.groups_to_exclude, self.cols_to_exclude = self._build_exclusion_sets()

        self._get_low_prevalence_cols()

        # Zwei Varianten: nur Hauptdiagnose (pdx) und Haupt- + Nebendiagnosen (pdx_sdx), da ICD-Granularitätsvergleich für beide Konfigurationen separat durchgeführt wird
        # Ausschließliche Verwendung der Hauptdiagnose
        self._process_pdx()
        # Verwendung der Haupt- und Nebendiagnosen
        self._process_pdx_sdx()

        # als Feather Dateien speichern
        self.res_pdx.to_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "grouped_campus_file_pdx_typ_vwd.feather"))
        self.res_pdx_sdx.to_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "grouped_campus_file_pdx_sdx_typ_vwd.feather"))

        # für Mortalität bearbeiten (erfordert ausgewogengenheit der Codes bzgl. Mortalität)
        self._get_unbalanced_mortality_cols()
        self._process_pdx(process_mortality=True)
        self._process_pdx_sdx(process_mortality=True)
        
        # als Feather Dateien speichern
        self.res_pdx.to_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "grouped_campus_file_pdx_entl_grd.feather"))
        self.res_pdx_sdx.to_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "grouped_campus_file_pdx_sdx_entl_grd.feather"))
        self.logger.info("Speichern abgeschlossen.")


    def _cleanup_data(self):
        """
        Wrapper Funktion zum Aufbereiten der Daten.
        Erzwingt Datentyp, entfernt nicht vorhandene Werte, erzwingt richtige Werte, entfernt und dokumentiert Fehler DRG-Codes, speichert die fertigen Daten.
        """
        self._enforce_dtype()
        self._drop_nan()
        self._monitor_true_values()
        self._monitor_error_drg()
        self._save_ungrouped_df()

    def process_data(self):
        """
        Hauptfunktion zur Verarbeitung des Campus Files.
        Lädt das Campus Files, bereitet diese auf und speichert sie wieder.
        """
        self.logger.info("Verarbeite das Campus File.")
        self.df = self._load_campus_file()

        # Gruppen laden
        self.icd_groups = self._load_cleaned_icd_groups()
        self.ops_groups = self._load_cleaned_ops_groups()
        self.drg_codes = self._load_cleaned_drg_codes()

        # Metadaten laden
        self.columns_metadata = self._load_columns_metadata()

        # Daten bereinigen
        self._cleanup_data()

        # Daten speichern
        self._save_grouped_df_by_icd()

        self.logger.info("Verarbeitung des Campus Files abgeschlossen.")

    def load_ungrouped_dataframe(self):
        """
        Funktion zum Laden des bereinigten Campus Files ohne Multi-Hot-Konvertierung.
        """
        self.logger.info("Lade Feather Daten")
        # Campus File laden
        self.df = pd.read_feather(os.path.join(DATA_DIR, "Campus_File_DRG_2010", "ungrouped_df.feather"))

        # Gruppen laden
        self.icd_groups = self._load_cleaned_icd_groups()
        self.ops_groups = self._load_cleaned_ops_groups()
        self.drg_codes = self._load_cleaned_drg_codes()

        # Metadaten laden
        self.columns_metadata = self._load_columns_metadata()
        self.logger.info("Ungruppiertes Campus File geladen")
        