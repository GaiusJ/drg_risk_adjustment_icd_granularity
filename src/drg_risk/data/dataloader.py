"""
Lädt und bereinigt die vorverarbeiteten Campus-File-Daten (Feather-Format) für das Modelltraining. Entfernt ICD-Codes mit geringer Prävalenz, unausgeglichener Mortalität und diagnosefremden Kapiteln.
"""

import pandas as pd

from drg_risk.utils.logger import Logger

class DataLoader():
    """
    Klasse zum Laden der Daten während des Trainings.
    """
    def __init__(self, feather_path_pdx, feather_path_pdx_sdx):
        """
        Initialisiert den DataLoader mit den Pfaden zu den Feather-Dateien und den Metadaten.
        Args:
            feather_path_pdx: Pfad zur Feather-Datei (nur Hauptdiagnosen)
            feather_path_pdx_sdx: Pfad zur Feather-Datei (Haupt- und Nebendiagnosen)
        """
        self.feather_path_pdx =     feather_path_pdx
        self.feather_path_pdx_sdx = feather_path_pdx_sdx
        self.logger =               Logger(name = "dataloader")

        self.logger.info("Dataloader erfolgreich initialisiert.")


    def load_data(self, output_var):
        """
        Lädt das Campus File aus Feather Daten.

        Args:
            output_var: Zielvariable ('entl_grd' für Mortalität, 'typ_vwd' für Verweildauer)
        """
        self.logger.info("Lade Daten")
        self.df_pdx =       pd.read_feather(f"{self.feather_path_pdx}_{output_var}.feather")
        self.df_pdx_sdx =   pd.read_feather(f"{self.feather_path_pdx_sdx}_{output_var}.feather")

        self.logger.info("Daten erfolgreich geladen.")

    def get_data(self, use_sdx:bool = True, columns = None, include_icd_groups = True, include_icd_chapter = True, subset:dict = None, shuffle = True, seed = 42):
        """
        Gibt aufbereitete Trainingsdaten zurück, optional gefiltert und gemischt.

        Args:
            use_sdx: Wenn True, Haupt- und Nebendiagnosen; sonst nur Hauptdiagnosen
            columns: Basisspalten die zurückgegeben werden (ICD-Spalten werden automatisch ergänzt)
            include_icd_groups: ICD-Gruppen-Spalten einschließen
            include_icd_chapter: ICD-Kapitel-Spalten einschließen
            subset: Dict zur Filterung, z.B. {'aufn_anl': ['N']} für Notfälle
            shuffle: Zufällige Reihenfolge der Zeilen
            seed: Seed für Reproduzierbarkeit
        Returns:
            DataFrame mit Trainings- und Zielvariablen
        """
        self.logger.info("Lade Daten.")
        self.logger.info(f"Verwendung von Nebendiagnosen: {use_sdx}.")

        # Daten kopieren zum Vermeiden von Verweisen und Überschreiben der Daten
        if use_sdx:
            data = self.df_pdx_sdx.copy()
        else:
            data = self.df_pdx.copy()

        if subset:
            self.logger.info(f"Verwende folgendes Subset: {subset}")
            for column, values in subset.items():
                data = data[data[column].isin(values)]

        # ICD-Spalten automatisch zur übergebenen columns-Liste hinzufügen, je nach include_icd_chapter/include_icd_groups
        if columns:
            self.logger.info(f"Ausgabe der folgenden Spalten: {columns}")
            for col in data.columns:
                self.logger.info(f"inkludiere Kapitel ICD Spalten: {include_icd_chapter}")
                if include_icd_chapter:
                    if "icd_chapter" in col:
                        columns.append(col)
                self.logger.info(f"inkludiere Gruppen ICD Spalten: {include_icd_groups}")
                if include_icd_groups:
                    if "icd_group" in col:
                        columns.append(col)
            data = data[columns]

        self.logger.info(f"Daten werden gemischt: {shuffle}")
        if shuffle:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
            self.logger.info(f"Daten wurden gemischt mit seed: {seed}")
        self.logger.info("Daten zurückgegeben.")
        return data