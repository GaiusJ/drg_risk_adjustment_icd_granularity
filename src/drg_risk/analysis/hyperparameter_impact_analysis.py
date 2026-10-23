"""
Haupteffekt- und Interaktionsanalyse der Modell-Hyperparameter.
Lädt AUROC-Ergebnisse aller 240 Modellkonfigurationen und erzeugt Typst-Tabellen für Haupt- und Interaktionseffekte (Mortalität und Verweildauer).
"""

import itertools
import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats import anova as anova_stats

from drg_risk import REPORT_DIR, PROJECT_ROOT

# 240 Kombinationen aus ICD-Struktur, Subgruppe, Input-Variable, Nebendiagnosen und Kovarianten
icd_structures = ["group", "chapter"]
output_vars = ["typ_vwd", "entl_grd"]
subsets = [{"aufn_anl": ["N"]}, {"aufn_anl": ["E", "V", "G"]}, {"partition": ["M"]}, {"partition": ["O"]}, None]
input_vars = ["codes", "charlson", "elixhauser"]
use_sdx_ = [True, False]
kovarianten_usage = [["typ_alter", "sex", "aufn_anl", "anzahl_nebendiagnosen"], None]

combinations = list(itertools.product(
    icd_structures, subsets, input_vars, use_sdx_, kovarianten_usage, output_vars
))

# zur Umbenennung
FAKTOR_LABELS = {
    "C(icd_structure)": "ICD-Struktur",
    "C(subset)": "Subgruppe",
    "C(input_var)": "Unabhängige Variable",
    "C(use_sdx)": "Verwendung Nebendiagnosen",
    "C(kovarianten)": "Verwendung Kovarianten",
    "C(icd_structure):C(subset)": "ICD-Struktur × Subgruppe",
    "C(icd_structure):C(input_var)": "ICD-Struktur × Unabhängige Variable",
    "C(icd_structure):C(use_sdx)": "ICD-Struktur × Nebendiagnosen",
    "C(icd_structure):C(kovarianten)": "ICD-Struktur × Kovarianten",
    "C(subset):C(input_var)": "Subgruppe × Unabhängige Variable",
    "C(subset):C(use_sdx)": "Subgruppe × Nebendiagnosen",
    "C(subset):C(kovarianten)": "Subgruppe × Kovarianten",
    "C(input_var):C(use_sdx)": "Unabhängige Variable × Nebendiagnosen",
    "C(input_var):C(kovarianten)": "Unabhängige Variable × Kovarianten",
    "C(use_sdx):C(kovarianten)": "Nebendiagnosen × Kovarianten",
}



def _load_data():
    """
    Lädt AUROC-Metriken aller Modellkonfigurationen aus den HTML-Berichten.

    Returns:
        DataFrame mit einer Zeile pro Modellkonfiguration und Spalten für alle Hyperparameter sowie AUROC, Laufzeit und Fallzahl.
    """
    results = []

    for icd_struct, subset, input_var, use_sdx, kovarianten, output_var in combinations:
        # Aus dem subset-Dict den Pfad-Substring für den Report-Ordnernamen bauen
        if subset:
            key = list(subset.keys())[0]
            value = list(subset.values())[0]
            value_str = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            subset_str = f"subset {key}={value_str}"
        else:
            subset_str = "subset=gesamt"

        # Pfad definieren
        path_dir = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={output_var}", subset_str, f"input_var={input_var}", f"sdx={use_sdx}")
        kovarianten_bool = bool(kovarianten)
        full_report_path = os.path.join(path_dir, f"{icd_struct}_kovarianten={kovarianten_bool}.html")

        # Bericht einlesen
        tabellen = pd.read_html(full_report_path)

        auroc = tabellen[0].iloc[1]["mean"]

        with open(full_report_path, "r", encoding="utf-8") as f:
            html_text = f.read()

        # Werte aus dem rohen HTML-Text parsen, da sie nicht in den pd.read_html-Tabellen stehen (Freitext im Report)
        time = float(html_text.split("</table>")[-1].split(" ")[-1])
        features = int(html_text.split(" Features")[0].split(" ")[-1])
        training_samples = int(html_text.split("Trainingsdaten: ")[1].split(" ")[0])

        results.append({
            "icd_structure": icd_struct,
            "subset": subset_str,
            "input_var": input_var,
            "use_sdx": use_sdx,
            "kovarianten": kovarianten_bool,
            "output_var": output_var,
            "AUROC": auroc,
            "Zeit [s]": time,
            "feature": features,
            "Anzahl Krankenhausfälle": training_samples
        })

    results_df = pd.DataFrame(results)

    return results_df

def _format_typst_main_effects(df_mort, df_vwd, factors, metric="AUROC"):
    """
    Berechnet die mittleren AUROC-Haupteffekte je Hyperparameter-Stufe und formatiert sie als Typst-Tabellencode (String).

    Args:
        df_mort: DataFrame mit Mortalitätsergebnissen
        df_vwd: DataFrame mit Verweildauerergebnissen
        factors: Liste der Hyperparameter
        metric: Zielmetrik, Standard AUROC
    Returns:
        Typst-Code als String
    """
    lines = []
    lines.append("#figure(")
    lines.append("  caption: [Haupteffekte der Hyperparameter],")
    lines.append("  table(")
    lines.append("    columns: (auto, auto, auto, auto),")
    lines.append("    align: (left, left, right, right),")
    lines.append("    table.header(")
    lines.append("      [*Faktor*], [*Stufe*], [*Mortalität*], [*Verweildauer*],")
    lines.append("    ),")

    # Haupteffekte berechnen
    for factor in factors:
        # für Vergleichbarkeit ausschließen, wenn ICD-Struktur ein betrachteter Faktor ist
        # if factor == "icd_structure":
        #     df_vwd_temp = df_vwd[~(df_vwd["input_var"].isin(["elixhauser", "charlson"]) & (df_vwd["icd_structure"] == "group"))].copy()
        #     df_vwd_mort = df_mort[~(df_mort["input_var"].isin(["elixhauser", "charlson"]) & (df_mort["icd_structure"] == "group"))].copy()
        # else:
        df_vwd_temp = df_vwd
        df_vwd_mort = df_mort
        mort = df_vwd_mort.groupby(factor)[metric].mean().round(3)
        vwd = df_vwd_temp.groupby(factor)[metric].mean().round(3)

        all_levels = sorted(set(mort.index) | set(vwd.index))

        for i, level in enumerate(all_levels):
            factor_cell = f"[{factor}]" if i == 0 else "[]"
            mort_val = f"{mort.loc[level]:.3f}" if level in mort.index else "---"
            vwd_val = f"{vwd.loc[level]:.3f}" if level in vwd.index else "---"
            lines.append(f"    {factor_cell}, [{level}], [{mort_val}], [{vwd_val}],")

        # Delta berechnen
        if factor == "icd_structure":
            mort_delta = mort.loc["group"] - mort.loc["chapter"] if "group" in mort.index and "chapter" in mort.index else None
            vwd_delta = vwd.loc["group"] - vwd.loc["chapter"] if "group" in vwd.index and "chapter" in vwd.index else None
        else:
            mort_delta = mort.max() - mort.min()
            vwd_delta = vwd.max() - vwd.min()

        mort_d = f"{mort_delta:.3f}" if mort_delta is not None else "---"
        vwd_d = f"{vwd_delta:.3f}" if vwd_delta is not None else "---"
        lines.append(f"    [], [*Δ*], [*{mort_d}*], [*{vwd_d}*],")

    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def _format_typst_interaction(df, factor_1, factor_2, metric="AUROC"):
    """
    Berechnet Interaktionseffekte zwischen zwei Hyperparametern und formatiert sie als Typst-Tabellencode (String).

    Args:
        df: DataFrame (Mortalität oder Verweildauer)
        factor_1, factor_2: die zwei zu kreuzenden Hyperparameter
        metric: Zielmetrik, Standard AUROC
    Returns:
        Typst-Code als String
    """

    # für Vergleichbarkeit ausschließen, wenn ICD-Struktur ein betrachteter Faktor ist
    # if factor_1 == "icd_structure" or factor_2 == "icd_structure":
    #     df_temp = df[~(df["input_var"].isin(["elixhauser", "charlson"]) & (df["icd_structure"] == "group"))].copy()
    # else:
    df_temp = df

    # Interaktionseffekte berechnen
    
    table = df_temp.groupby([factor_1, factor_2])[metric].mean().round(3).unstack()
    delta = table.max(axis=1) - table.min(axis=1)
    table["Δ"] = delta.round(3)

    delta_row = table.apply(lambda col: (col.max() - col.min())).round(3)
    table.loc["Δ"] = delta_row

    cols = list(table.columns)
    n_cols = len(cols) + 1  # +1 für Zeilenlabel

    # in typst Tabelle umwandeln
    lines = []
    lines.append(f"===== {factor_1} × {factor_2}")
    lines.append("")
    lines.append("#figure(")
    lines.append(f"  caption: [{factor_1} × {factor_2}],")
    lines.append(f"  table(")
    lines.append(f"    columns: ({', '.join(['auto'] * n_cols)}),")
    lines.append(f"    align: (left, {', '.join(['center'] + ['right'] * (n_cols - 1))}),")
    lines.append(f"    table.header(")
    lines.append(f"      table.cell(colspan: 1)[], table.cell(colspan: {n_cols - 2})[*{factor_2}*], table.cell(colspan: 1)[],")
    lines.append(f"      [*{factor_1}*], " + ", ".join(f"[*{c}*]" for c in cols) + ",")
    lines.append(f"    ),")

    for idx, row in table.iterrows():
        is_delta_row = idx == "Δ"
        cells = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                cells.append("[]")
            elif is_delta_row or c == "Δ":
                cells.append(f"[*{val:.3f}*]")
            else:
                cells.append(f"[{val:.3f}]")

        label = f"[*{idx}*]" if is_delta_row else f"[{idx}]"
        lines.append(f"    {label}, {', '.join(cells)},")

    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def _format_typst_interactions(df_mort, df_vwd, factors, metric="AUROC"):
    """
    Iteriert über alle Hyperparameter-Paare und gibt den kombinierten Typst-Code für Mortalität und Verweildauer zurück.
    """
    lines = []
    for f1, f2 in itertools.combinations(factors, 2):
        lines.append("==== Mortalität")
        lines.append(_format_typst_interaction(df_mort, f1, f2, metric))
        lines.append("")
        lines.append("==== Verweildauer")
        lines.append(_format_typst_interaction(df_vwd, f1, f2, metric))
        lines.append("")
        lines.append("")
    return "\n".join(lines)

def compute_omega_sq(df_mort, df_vwd, score="AUROC", interactions = False):
    """
    Berechnet die erklärte Varianz der Hyperparameter mittels Anova und ω²-Analyse.
    """
    if interactions:
        factors = "(C(icd_structure) + C(subset) + C(input_var) + C(use_sdx) + C(kovarianten))**2"
    else:
        factors = "C(icd_structure) + C(subset) + C(input_var) + C(use_sdx) + C(kovarianten)"
    results = {}

    for output_var, df in [("Mortalität", df_mort), ("Verweildauer", df_vwd)]:
        model = smf.ols(f"{score} ~ {factors}", data=df).fit()
        aov = anova_stats.anova_lm(model, typ=2)

        ss_residual = aov.loc["Residual", "sum_sq"]
        df_residual = aov.loc["Residual", "df"]
        ms_residual = ss_residual / df_residual
        ss_total = aov["sum_sq"].sum()

        # Discovering Statistics Using IBM SPSS Statistics“ (5. Auflage 2018) von Andy Field, Gleichung 12.32
        aov["omega_sq"] = (aov["sum_sq"] - (aov["df"] * ms_residual)) / (ss_total + ms_residual)
        aov["omega_sq"] = aov["omega_sq"].clip(lower=0)

        aov_clean = aov.loc[aov.index != "Residual", ["omega_sq"]].copy()
        aov_clean.index = aov_clean.index.map(lambda x: FAKTOR_LABELS.get(x, x))
        aov_clean = aov_clean.sort_values("omega_sq", ascending=False)
        aov_clean = aov_clean.rename(columns={"omega_sq": "ω²"})

        results[output_var] = aov_clean

    return results

def _format_typst_omega_sq(results: dict) -> str:
    """
    Formatiert Omega²-Ergebnisse aus compute_omega_sq als Typst-Tabellencode.

    Args:
        results: Dict mit Keys "Mortalität" und "Verweildauer", Values sind DataFrames mit Index=Faktor und Spalte "ω²"
    Returns:
        Typst-Code als String
    """
    df_mort = results["Mortalität"]
    df_vwd = results["Verweildauer"]

    all_factors = list(dict.fromkeys(list(df_mort.index) + list(df_vwd.index)))

    lines = []
    lines.append("#figure(")
    lines.append("  caption: [Erklärte Varianz der Hyperparameter (ω²)],")
    lines.append("  table(")
    lines.append("    columns: (auto, auto, auto),")
    lines.append("    align: (left, right, right),")
    lines.append("    table.header(")
    lines.append("      [*Faktor*], [*Mortalität*], [*Verweildauer*],")
    lines.append("    ),")

    for factor in all_factors:
        mort_val = df_mort.loc[factor, "ω²"] if factor in df_mort.index else None
        vwd_val = df_vwd.loc[factor, "ω²"] if factor in df_vwd.index else None
        mort_str = f"{mort_val:.3f}" if mort_val is not None else "---"
        vwd_str = f"{vwd_val:.3f}" if vwd_val is not None else "---"
        lines.append(f"    [{factor}], [{mort_str}], [{vwd_str}],")

    lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def calc_hyperparameter_impact():
    """
    Wrapper Funktion zum Berechnen des Einflusses der Hyperparameter und Umwandlung in typst Tabellen.
    """
    results_df = _load_data()

    df_mort = results_df[results_df["output_var"] == "entl_grd"].copy()
    df_vwd = results_df[results_df["output_var"] == "typ_vwd"].copy()

    hyperparams = ["icd_structure", "subset", "input_var", "use_sdx", "kovarianten"]

    # Technische Bezeichner in lesbare deutsche Labels übersetzen für die Thesis-Tabellen
    main_effects = _format_typst_main_effects(df_mort, df_vwd, hyperparams).replace("icd_structure", "ICD-Struktur").replace("chapter", "Kapitel").replace("group", "Gruppe").replace("subset", "Subgruppe").replace("input_var", "unabhängige Variable").replace("use_sdx", "Verwendung Nebendiagnosen").replace("kovarianten", "Verwendung Kovarianten").replace("True", "Ja").replace("False", "Nein").replace("charlson", "Charlsonlike").replace("elixhauser", "Elixhauserlike").replace("codes", "Codes").replace("aufn_anl", "Aufnahmeanlass").replace("=E,V,G", " ist kein Notfall").replace("=N", " ist ein Notfall").replace("partition", "Partition ist ").replace("=M", "medizinisch").replace("=O", "operativ").replace("=gesamt", " vollständig")
    interactions = _format_typst_interactions(df_mort, df_vwd, hyperparams).replace("icd_structure", "ICD-Struktur").replace("chapter", "Kapitel").replace("group", "Gruppe").replace("subset", "Subgruppe").replace("input_var", "unabhängige Variable").replace("use_sdx", "Verwendung Nebendiagnosen").replace("kovarianten", "Verwendung Kovarianten").replace("True", "Ja").replace("False", "Nein").replace("charlson", "Charlsonlike").replace("elixhauser", "Elixhauserlike").replace("codes", "Codes").replace("aufn_anl", "Aufnahmeanlass").replace("=E,V,G", " ist kein Notfall").replace("=N", " ist ein Notfall").replace("partition", "Partition ist ").replace("=M", "medizinisch").replace("=O", "operativ").replace("=gesamt", " vollständig")

    omega_haupteffekte = compute_omega_sq(df_mort, df_vwd, interactions=False)
    omega_interaktionen = compute_omega_sq(df_mort, df_vwd, interactions=True)

    omega_typ = _format_typst_omega_sq(omega_haupteffekte)
    omega_interaktion_typ = _format_typst_omega_sq(omega_interaktionen)
    # Datei speichern
    with open(os.path.join(PROJECT_ROOT, "Ausarbeitung", "auswertungen", "hyperparameter_auswertung.typ"), "w", encoding="utf-8") as f:
        # Regex ersetzt Dezimalpunkt durch Komma für deutsche Zahlenschreibweise in Typst
        f.write("""#show regex("(^|[^0])([0])\.([0-9]+)"): it => {
                    it.text.replace(".", ",")
                }
            #page(flipped: true)[""")
        f.write("== Einflussananalyse der Hyperparameter\n\n")
        f.write("=== Haupteffekte\n\n")
        f.write(main_effects)
        f.write(omega_typ)
        f.write("\n\n=== Interaktionseffekte\n\n")
        f.write(interactions)
        f.write(omega_interaktion_typ)

        f.write("]")

def time_analysis():
    """
    Berechnet den Einfluss der Hyperparameter auf die Laufzeit.
    """
    results_df = _load_data()

    results_df_mort = results_df[results_df["output_var"] == "entl_grd"]
    results_df_vwd = results_df[results_df["output_var"] == "typ_vwd"]

    print("=== Mortalität ===")

    print(results_df_mort.groupby("subset")[["Anzahl Krankenhausfälle", "Zeit [s]"]].mean().astype(int))

    print(f"Mittelwert Zeit [s]: {results_df_mort['Zeit [s]'].mean():.0f}")

    print(f"Standardabweichung Zeit [s]: {results_df_mort['Zeit [s]'].std():.0f}")

    print("\n=== Verweildauer ===")

    print(results_df_vwd.groupby("subset")[["Anzahl Krankenhausfälle", "Zeit [s]"]].mean().astype(int))

    print(f"Mittelwert Zeit [s]: {results_df_vwd['Zeit [s]'].mean():.0f}")

    print(f"Standardabweichung Zeit [s]: {results_df_vwd['Zeit [s]'].std():.0f}")