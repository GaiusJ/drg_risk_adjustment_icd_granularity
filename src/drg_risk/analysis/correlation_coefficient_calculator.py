"""
Replikationsanalyse der Koeffizientenstruktur aus Stausberg & Hagn (2015).
Vergleicht ICD-Koeffizienten des eigenen Modells mit der Referenzstudie auf Kapitel- und Gruppenebene mittels Pearson-Korrelation.
"""

import os
import pandas as pd
from docx import Document
from scipy.stats import pearsonr

from drg_risk import REPORT_DIR, PROJECT_ROOT


# Hyperparameter des Baseline Modells
output_var = "entl_grd"
subset = "gesamt"
input_var = "codes"
sdx = True
kovarianten = False # keine Kovarianten verwenden, damit gleiche Hyperparameter, wie Stausberg und Hagn

def _obtain_own_coefficients(icd_structure):
    """
    Lädt die ICD-Koeffizienten des eigenen Baseline-Modells (Bamberg) aus der postprocessten HTML-Ergebnisdatei.

    Args:
        icd_structure: 'chapter' oder 'group'
    Returns:
        DataFrame mit Koeffizienten des eigenen Modells
    """

    # Koeffizienten laden
    path = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={output_var}", f"subset={subset}", f"input_var={input_var}", f"sdx={sdx}", f"{icd_structure}_kovarianten={kovarianten}.html")
    coefficient_table = pd.read_html(path)[-1]

    # Tabelle aufbereiten
    coefficient_table = coefficient_table[["name", "coefficient"]]
    # Nur Zeilen behalten, die ICD-Koeffizienten sind
    coefficient_table = coefficient_table[[name.startswith(f"icd_{icd_structure}") for name in coefficient_table["name"]]]
    coefficient_table['name'] = coefficient_table['name'].str.replace(f"icd_{icd_structure}_is_", '', regex=False)
    coefficient_table = coefficient_table.rename(columns = {"name": "Name_Bamberg", "coefficient": "Koeffizient_Bamberg"})

    return coefficient_table

def _obtain_stausberg_hagn_coefficients(icd_structure):
    """
    Extrahiert ICD-Koeffizienten aus der digitalisierten Tabelle von Stausberg & Hagn (2015) als Referenzmodell für den Replikationsvergleich.

    Args:
        icd_structure: 'chapter' oder 'group'
    Returns:
        DataFrame mit Spalten [ICD-10-{icd_structure}, Koeffizient_Stausberg_Hagn, Inpatients]
    """

    # Koeffizienten laden
    path = os.path.join(PROJECT_ROOT, "Ausarbeitung", "Stausberg & Hagn", f"Auswertung_Stausberg_Hagn_{icd_structure}.docx")
    doc = Document(path)
    coefficient_table = doc.tables[-1]

    # Tabelle aufbereiten
    data = []
    for row in coefficient_table.rows:
        data.append([cell.text.strip() for cell in row.cells])

    # Erste Zeile (Kopfzeile-Duplikat) und letzte Zeile (Intercept) entfernen
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.iloc[1:-1]

    df["Inpatients"] = df["Inpatients"].str.replace(",", "").astype(int)
    # nur ICD-Code und Koeffizienten behalten
    df = pd.concat([df.iloc[:, :1], df.iloc[:, 2:]], axis=1)

    df = df.rename(columns = {"ICD-10-group": f"ICD-10-{icd_structure}", "Coefficient β": "Koeffizient_Stausberg_Hagn"})

    return df

def _combine_coefficients(df_bamberg, df_stausberg_hagn, icd_structure):
    """
    Fasst die Koeffizienten von Bamberg und Stausberg & Hagn in eine Tabelle zusammen.
    """

    # Outer Join: ICD-Codes die nur in einem Modell vorkommen erhalten Koeffizient 0
    return pd.merge(df_bamberg, df_stausberg_hagn, left_on="Name_Bamberg", right_on=f"ICD-10-{icd_structure}", how = "outer").fillna(0)

def _calc_corr_p_values(merged_df, column1, column2):
    """
    Berechnet den Pearson Korrelationskoeffizient und den zugehörigen p-Wert.
    """
    df_clean = merged_df[[column1, column2]].dropna()

    x = df_clean[column1].astype(float).to_numpy()
    y = df_clean[column2].astype(float).to_numpy()

    pearson, p_val = pearsonr(x, y)

    return pearson, p_val

def _roman_to_int(s):
    """
    Konvertiert römische Ziffern in arabische Ganzzahlen als String.
    Nötig weil Stausberg & Hagn ICD-Kapitel als römische Ziffern ausweisen (z.B. 'IV'), das eigene Modell jedoch arabische Ziffern verwendet.

    Args:
        s: Römische Ziffer als String
    Returns:
        Arabische Ziffer als String
    """

    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    total = 0
    previous_val = 0

    for char in reversed(s.upper().strip()):
        value = roman_map.get(char, 0)

        # wenn aktueller Wert größer gleich dem vorherigen Wert addieren
        if value >= previous_val:
            total += value
        # wenn aktueller Wert kleiner ist (z. B. IV): subtrahieren
        else:
            total -= value
        previous_val = value

    return str(total)

def calc_correlation():
    """
    Berechnet die Korrelation der Größe der Koeffizienten von Bamberg und Stausberg & Hagn.
    """

    icd_structure = "group"
    # Daten laden
    df_bamberg_group = _obtain_own_coefficients(icd_structure)
    df_stausberg_hagn_group = _obtain_stausberg_hagn_coefficients(icd_structure)

    # Koeffizienten zusammenfügen
    df_merged_group = _combine_coefficients(df_bamberg_group, df_stausberg_hagn_group, icd_structure)

    # p-Wert und Korrelationskoeffizient berechnen
    pearson_group, p_val_group = _calc_corr_p_values(df_merged_group, "Koeffizient_Bamberg", "Koeffizient_Stausberg_Hagn")

    # Gruppen ausschließen, die nicht in beiden Modellen vorhanden sind, zum Vergleich
    df_merged_non_zero_group = df_merged_group.copy()
    df_merged_non_zero_group = df_merged_non_zero_group[df_merged_non_zero_group["Koeffizient_Bamberg"] != 0]
    df_merged_non_zero_group = df_merged_non_zero_group[df_merged_non_zero_group["Koeffizient_Stausberg_Hagn"] != 0]

    # p-Wert und Korrelationskoeffizient berechnen
    pearson_group_non_zero, p_val_group_non_zero = _calc_corr_p_values(df_merged_non_zero_group, "Koeffizient_Bamberg", "Koeffizient_Stausberg_Hagn")

    print(f"Pearson Korrelationskoeffizient Gruppe: {pearson_group} (p = {p_val_group}), N = {len(df_merged_group)}")
    print(f"Pearson Korrelationskoeffizient Gruppe (Koeffizienten = 0 ausgeschlossen): {pearson_group_non_zero} (p = {p_val_group_non_zero}), N = {len(df_merged_non_zero_group)}")


    icd_structure = "chapter"
    # Daten laden
    df_bamberg_chapter = _obtain_own_coefficients(icd_structure)
    df_stausberg_hagn_chapter = _obtain_stausberg_hagn_coefficients(icd_structure)

    # Kapitel-IDs von römisch nach arabisch konvertieren (Stausberg & Hagn verwenden römische Ziffern)
    df_stausberg_hagn_chapter["ICD-10-chapter"] = [_roman_to_int(chapter) for chapter in df_stausberg_hagn_chapter["ICD-10-chapter"]]

    # Koeffizienten zusammenfügen
    df_merged_chapter = _combine_coefficients(df_bamberg_chapter, df_stausberg_hagn_chapter, icd_structure)

    # p-Wert und Korrelationskoeffizient berechnen
    pearson_chapter, p_val_chapter = _calc_corr_p_values(df_merged_chapter, "Koeffizient_Bamberg", "Koeffizient_Stausberg_Hagn")

    # Gruppen ausschließen, die nicht in beiden Modellen vorhanden sind, zum Vergleich
    df_merged_non_zero_chapter = df_merged_chapter.copy()
    df_merged_non_zero_chapter = df_merged_non_zero_chapter[df_merged_non_zero_chapter["Koeffizient_Bamberg"] != 0]
    df_merged_non_zero_chapter = df_merged_non_zero_chapter[df_merged_non_zero_chapter["Koeffizient_Stausberg_Hagn"] != 0]

    # p-Wert und Korrelationskoeffizient berechnen
    pearson_chapter_non_zero, p_val_chapter_non_zero = _calc_corr_p_values(df_merged_non_zero_chapter, "Koeffizient_Bamberg", "Koeffizient_Stausberg_Hagn")

    print(f"Pearson Korrelationskoeffizient Kapitel: {pearson_chapter} (p = {p_val_chapter}), N = {len(df_merged_chapter)}")
    print(f"Pearson Korrelationskoeffizient Kapitel (Koeffizienten = 0 ausgeschlossen): {pearson_chapter_non_zero} (p = {p_val_chapter_non_zero}), N = {len(df_merged_non_zero_chapter)}")