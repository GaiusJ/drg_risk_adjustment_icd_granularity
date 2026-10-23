"""
Erstellt eine statistische Zusammenfassung des Campus Files 2010 und exportiert sie als Typst-Dokument.
"""

from drg_risk.data.datapreprocessor import DataPreprocessor
from drg_risk import DATA_DIR, PROJECT_ROOT
from drg_risk.utils.logger import Logger

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CFSummarizer():
    """
    Klasse für eine statistische Zusammenfassung des Campus Files.
    Lädt das Campus File und erstellt dann einen Bericht der statistischen Zusammenfassung des Campus Files.
    """
    def __init__(self):
        self.logger = Logger("CF Zusammenfassung")
        self.logger.info("Daten werden geladen.")
        preprocessor = DataPreprocessor(
            data_path =                 os.path.join(DATA_DIR, "Campus_File_DRG_2010", "Campus_File_DRG_2010_DRG_Statistik.csv"),
            icd_sys_path =              os.path.join(DATA_DIR, "icd_systematik_2010.json"),
            ops_sys_path =              os.path.join(DATA_DIR, "ops_systematik_2010.json"),
            columns_metadata_path =     os.path.join(DATA_DIR, "Campus_File_DRG_2010", "columns_metadata.json"),
            subgroups_path =            os.path.join(DATA_DIR, "icd_subgruppen.json"),
            drg_path =                  os.path.join(DATA_DIR, "Fallpauschalenkatalog_2010.xls"),
        )
        # Lädt Campus File
        preprocessor.load_ungrouped_dataframe()
        self.df = preprocessor.df

        self.logger.info("Campus File geladen.")

    def _to_typst_formatted(self, df):
        """
        Formatiert ein DataFrame zu einer Typst Tabelle.
        Hilfsmethode identisch mit StabilityAnalyzer._to_typst_formatted
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

    def generate_report(self):
        """
        Erstellt den Bericht für die statistische Zusammenfassung des Campus Files.
        """
        # Metriken berechnen
        self._get_scalar_values()
        self._get_dataframes()

        self.logger.info("Erstelle Zusammenfassungs Report.")

        # Aufbereitung der Altermetriken
        # Klassengrenzen der ordinalen Altersvariable
        age_data = {
            'typ_alter': self.altersverteilung["typ_alter"],
            'Anzahl': self.altersverteilung["Anzahl"],
            'unten': [0, 1, 10, 20, 30, 40, 50, 60, 70, 80],
            'oben': [1, 10, 20, 30, 40, 50, 60, 70, 80, 100]
        }
        df_age = pd.DataFrame(age_data)

        df_age['mitte'] = (df_age['unten'] + df_age['oben']) / 2
        df_age['breite'] = df_age['oben'] - df_age['unten']

        mean_age = (df_age['mitte'] * df_age['Anzahl']).sum() / self.fallzahl

        var_midpoint_biased_age = (df_age['Anzahl'] * (df_age['mitte'] - mean_age)**2).sum() / self.fallzahl
        std_uniform_age = np.sqrt(var_midpoint_biased_age)

        # Verweildauer
        data_vwd = {
            'typ_vwd': self.verweildauer_verteilung["typ_vwd"],
            'Anzahl': self.verweildauer_verteilung["Anzahl"]
        }
        df_vwd = pd.DataFrame(data_vwd)

        df_vwd['unten'] = [0,  1, 4,  8, 11, 15, 22]
        df_vwd['oben'] = [1,  4, 8, 11, 15, 22, 42]

        df_vwd['mitte'] = (df_vwd['unten'] + df_vwd['oben']) / 2
        df_vwd['breite'] = df_vwd['oben'] - df_vwd['unten']

        mean_vwd = (df_vwd['mitte'] * df_vwd['Anzahl']).sum() / self.fallzahl

        var_midpoint_vwd = (df_vwd['Anzahl'] * (df_vwd['mitte'] - mean_vwd)**2).sum() / self.fallzahl
        std_uniform_vwd = np.sqrt(var_midpoint_vwd)

        auswertungsstring = f"""
            #show regex("[\d_]{{2,}}_[\d_]+|\d+\.\d+"): it => {{
              if "_" in it.text {{
                it.text.replace("_", ".")
              }} else {{
                it.text.replace(".", ",")
              }}
            }}
            #show table.cell.where(y: 0): set text(weight: "bold")
            = Zusammenfassung des Campus Files

            == Skalare Werte

            Fallzahl: {self.fallzahl:_}; Sterbefälle: {self.sterbefälle:_}; Anteil: {self.sterbefälle/self.fallzahl*100:.3f} %

            Anzahl Nebendiagnosen: {(self.anzahl_nebendiagnosen_df['Durchschnitt'].iloc[0]):.3}±{(self.anzahl_nebendiagnosen_df['Standardabweichung'].iloc[0]):.3}

            Geschlechtereverteilung:
            Anteil weiblich: {(self.geschlechter_verteilung[self.geschlechter_verteilung["sex"] == "weiblich"]["Anzahl"].iloc[0] / self.fallzahl*100):.3} %; 
            Anteil männlich: {(self.geschlechter_verteilung[self.geschlechter_verteilung["sex"] == "männlich"]["Anzahl"].iloc[0] / self.fallzahl*100):.3} %

            Alter: {mean_age:.3}±{std_uniform_age:.3} Jahre

            == Verweildauer:
            
            Durchschnitt: {mean_vwd:.3}±{std_uniform_vwd:.3} Tage

            Verteilung:

            #figure(
                caption: [Verteilung der Verweildauer],
                image(
                    "{os.path.join("..", "Bilder", "verweildauer_verteilung.png").replace("\\", "/")}"
                )
            )<verweildauer_verteilung>

            == Aufnahmeanlass:
            {self._to_typst_formatted(
                self.aufnahmeanlass_verteilung.rename(columns={
                    "aufn_anl": "Aufnahmeanlass", 
                }).set_index("Aufnahmeanlass")
            )}

            == Top ICD Hauptdiagnosen:
            {self._to_typst_formatted(
                self.top_icd_hd.rename(columns={
                    "icd_hd3": "Hauptdiagnosegruppe", 
                }).set_index("Hauptdiagnosegruppe")
            )}

            == Top ICD Diagnosen:
            {self._to_typst_formatted(
                self.top_icd_hd_nd.rename(columns={
                    "0": "Diagnosegruppe", 
                }).set_index("Diagnosegruppe")
            )}

            == Partition:
            {self._to_typst_formatted(
                self.partition_verteilung.rename(columns={
                    "partition": "Partition", 
                }).set_index("Partition")
            )}
        """
        os.makedirs(os.path.join(PROJECT_ROOT, "Ausarbeitung", "Masterarbeit", "include"), exist_ok = True)
        path = os.path.join(PROJECT_ROOT, "Ausarbeitung", "Masterarbeit", "include", "campus_file_summary.typ")

        # Bericht schreiben
        with open(path, "w", encoding="utf-8") as f:
                f.write(auswertungsstring)
            
        self.logger.info("Report erstellt.")
    
    def _get_scalar_values(self):
        """
        Berechnet die Fallzahl und die Anzahl der Todesfälle.
        """
        self.logger.info("Berechne skalare Werte.")
        self.fallzahl = len(self.df)

        self.sterbefälle = sum(self.df["entl_grd"] == 5)

        self.logger.info("Skalare Werte berechnet.")


    def _get_dataframes(self):
        """
        Erstellt DataFrames für die statistische Auswertung von u.a. der Anzahl der Nebendiagnosen, dem Geschlecht, dem Alter und den Diagnosen.
        """
        self.logger.info("Berechne nicht skalare Werte.")

        nd_cols = [f"icd_nd{i}" for i in range(1, 11)]
        anzahl_nebendiagnosen = np.sum(~self.df[nd_cols].isna(), axis = 1)
        self.anzahl_nebendiagnosen_df = pd.DataFrame({
            "Durchschnitt": anzahl_nebendiagnosen.mean(),
            "IQR": anzahl_nebendiagnosen.quantile(.75) - anzahl_nebendiagnosen.quantile(.25),
            "Standardabweichung": np.std(anzahl_nebendiagnosen)
        }, index = ["Anzahl Nebendiagnosen"])

        geschlecht_rename_dict = {
            "m": "männlich", "w": "weiblich"
        }
        self.geschlechter_verteilung = (
            self.df["sex"]
            .map(geschlecht_rename_dict)
            .astype(pd.CategoricalDtype(categories=geschlecht_rename_dict.values(), ordered=True))
            .value_counts()
            .sort_index()
            .reset_index(name="Anzahl")
        )

        alter_rename_dict = {
            1: "< 1 Jahr alt",
            2: "1 bis < 10 Jahre",
            3: "10 bis < 20 Jahre",
            4: "20 bis < 30 Jahre",
            5: "30 bis < 40 Jahre",
            6: "40 bis < 50 Jahre",
            7: "50 bis < 60 Jahre",
            8: "60 bis < 70 Jahre",
            9: "70 bis < 80 Jahre",
            10: ">= 80 Jahre"
        }
        self.altersverteilung = (
            self.df["typ_alter"]
            .map(alter_rename_dict)
            .astype(pd.CategoricalDtype(categories=alter_rename_dict.values(), ordered=True))
            .value_counts()
            .sort_index()
            .reset_index(name="Anzahl")
        )
        self.altersverteilung["prozentualer Anteil [%]"] = self.altersverteilung["Anzahl"] / self.fallzahl * 100

        aufnahmeanlass_rename_dict = {
            "E": "Einweisung durch einen Arzt oder Zahnarzt",
            "N": "Notfall",
            "V": "Verlegung",
            "G": "Geburt"
        }
        self.aufnahmeanlass_verteilung = (
            self.df["aufn_anl"]
            .map(aufnahmeanlass_rename_dict)
            .astype(pd.CategoricalDtype(categories=aufnahmeanlass_rename_dict.values(), ordered=True))
            .value_counts()
            .sort_index()
            .reset_index(name="Anzahl")
        )
        self.aufnahmeanlass_verteilung["prozentualer Anteil [%]"] = self.aufnahmeanlass_verteilung["Anzahl"] / self.fallzahl * 100

        verweildauer_rename_dict = {
            1: "Stundenfall",
            2: "1 bis 3 Tage",
            3: "4 bis 7 Tage",
            4: "8 bis 10 Tage",
            5: "11 bis 14 Tage",
            6: "15 bis 21 Tage",
            7: "22 Tage oder länger"
        }
        self.verweildauer_verteilung = (
            self.df["typ_vwd"]
            .map(verweildauer_rename_dict)
            .astype(pd.CategoricalDtype(categories=verweildauer_rename_dict.values(), ordered=True))
            .value_counts()
            .sort_index()
            .reset_index(name="Anzahl")
        )
        self.verweildauer_verteilung["prozentualer Anteil"] = self.verweildauer_verteilung["Anzahl"] / self.fallzahl * 100

        os.makedirs(os.path.join(PROJECT_ROOT, "Ausarbeitung", "Masterarbeit", "Bilder"), exist_ok = True)

        # Verteilung als Bild
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.bar(self.verweildauer_verteilung["typ_vwd"], self.verweildauer_verteilung["prozentualer Anteil"])
        ax.set_xlabel("Verweildauer", fontsize=12)
        ax.set_ylabel("prozentualer Anteil", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.savefig(os.path.join(PROJECT_ROOT, "Ausarbeitung", "Masterarbeit", "Bilder", "verweildauer_verteilung.png"), bbox_inches="tight")
        plt.close()

        self.top_icd_hd = self.df["icd_hd3"].value_counts().reset_index(name="Anzahl").head()
        self.top_icd_hd["prozentualer Anteil [%]"] = self.top_icd_hd["Anzahl"] / self.fallzahl *100

        icd_cols = [col for col in self.df.columns if col.startswith("icd")]
        icd_list = self.df[icd_cols].values.ravel()
        icd_list = icd_list[pd.notna(icd_list)].tolist()
        self.top_icd_hd_nd = pd.DataFrame(icd_list).value_counts().head()
        self.top_icd_hd_nd = pd.DataFrame(self.top_icd_hd_nd).rename_axis("Diagnosegruppe").reset_index().rename(columns={"count": "Anzahl", "0": "Diagnose"})
        self.top_icd_hd_nd["prozentualer Anteil [%]"] = self.top_icd_hd_nd["Anzahl"] / self.fallzahl * 100

        partition_rename_dict = {
            "M": "medizinische Fallpauschalen",
            "O": "operative Fallpauschalen",
            "A": "andere Fallpauschalen, z. B. Koloskopie"
        }
        self.partition_verteilung = (
            self.df["partition"]
            .map(partition_rename_dict)
            .astype(pd.CategoricalDtype(categories=partition_rename_dict.values(), ordered=True))
            .value_counts()
            .sort_index()
            .reset_index(name="Anzahl")
        )
        self.partition_verteilung["prozentualer Anteil"] = self.partition_verteilung["Anzahl"] / self.fallzahl * 100

        self.logger.info("Nicht skalare Werte berechnet.")