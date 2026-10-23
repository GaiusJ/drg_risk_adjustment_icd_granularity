"""
Export von Modellmetriken und Regressionskoeffizienten als Typst-Tabellen für die Thesis-Ausarbeitung. Vergleicht Granularitätsstufen (Gruppe vs. Kapitel) und Modell vs. Baseline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from drg_risk import PROJECT_ROOT, REPORT_DIR, SCORES_DIR


class TypstTableExporter:
    """
    Klasse zum erstellen von Berichten anhand der berechneten Kennzahlen.
    """
    def __init__(self):
        self.AUSARBEITUNG_DIR = os.path.join(PROJECT_ROOT, "Ausarbeitung", "Auswertungen")

        self.icd_structures = ["group", "chapter"]
        self.output_vars = ["entl_grd", "typ_vwd"]

    def _convert_html_table_to_dataframe(self, path, output_var):
        """
        Konvertiert eine HTML-Tabelle in ein DataFrame.
        """
        tabellen = pd.read_html(path)

        new_df = tabellen[-1]

        new_df = new_df.drop(["p_value", "Unnamed: 0"], axis = 1)
        new_df = new_df.rename(columns = {"name": "Name", "coefficient": "Koeffizient", "odds_ratio": "OR", "std_error": "Standardfehler"})

        # Konfidenzintervall definieren
        new_df["Koeffizient KI"] = "[" + new_df["coef_CI_lower"].round(3).astype(str).str.replace(".", ",") + "; " + new_df["coef_CI_upper"].round(3).astype(str).str.replace(".", ",") + "]"
        new_df["OR KI"] = "[" + new_df["odds_CI_lower"].round(3).astype(str).str.replace(".", ",") + "; " + new_df["odds_CI_upper"].round(3).astype(str).str.replace(".", ",") + "]"

        new_df = new_df.drop(["coef_CI_lower", "coef_CI_upper"], axis = 1)
        new_df = new_df.drop(["odds_CI_lower", "odds_CI_upper"], axis = 1)

        new_df = new_df.loc[:, ["Name", "Koeffizient", "Koeffizient KI", "OR", "OR KI", "Standardfehler"]]

        # Spalten umbenennen
        new_df["Name"] = new_df["Name"].replace({
            "anzahl_nebendiagnosen": "Kovariante: Anzahl Nebendiagnosen",
            "aufn_anl_G": "Kovariante: Aufnahmeanlass = Geburt",
            "aufn_anl_N": "Kovariante: Aufnahmeanlass = Notfall",
            "aufn_anl_V": "Kovariante: Aufnahmeanlass = Verlegungen",
            "aufn_anl_E": "Kovariante: Aufnahmeanlass = Einweisung durch einen Arzt oder Zahnarzt",
            "intercepts": "Konstante",
            "sex": "Kovariante: Geschlecht = weiblich",
            "typ_alter": "Kovariante: Alter"
        })

        threshold_rename_dict = {}
        thresholds = [threshold for threshold in new_df["Name"] if "threshold" in threshold]
        thresholds = [threshold.split("_")[1] for threshold in thresholds]

        for threshold in thresholds:
            threshold_rename_dict[f"threshold_{threshold}"] = f"Schwellenwert {int(threshold) + 1}"

        chapter_rename_dict = {}
        chapters = [chapter for chapter in new_df["Name"] if "icd_chapter" in chapter]
        chapters = [chapter.split("_is_")[1] for chapter in chapters]

        for chapter in chapters:
            chapter_rename_dict[f"icd_chapter_is_{chapter}"] = f"ICD Kapitel = {chapter}"

        group_rename_dict = {}
        groups = [group for group in new_df["Name"] if "icd_group" in group]
        groups = [group.split("_is_")[1] for group in groups]

        for group in groups:
            group_rename_dict[f"icd_group_is_{group}"] = f"ICD Gruppe = {group}"
        
        new_df["Name"] = new_df["Name"].replace(threshold_rename_dict)
        new_df["Name"] = new_df["Name"].replace(chapter_rename_dict)
        new_df["Name"] = new_df["Name"].replace(group_rename_dict)

        # Nach Effektstärke sortieren (Betrag), unabhängig von Vorzeichen/Richtung
        new_df = new_df.iloc[new_df["Koeffizient"].abs().argsort()[::-1]]

        new_df = new_df.set_index(["Name"])

        # Konstanten ignorieren
        if output_var == "entl_grd":
            new_df = new_df.drop("Konstante")
        elif output_var == "typ_vwd":
            for i in range(1,7):
                new_df = new_df.drop(f"Schwellenwert {i}")

        return new_df.round(3)

    def _to_typst_formatted(self, df, cv_or_is):
            """
            Formatiert ein DataFrame zu einer Typst Tabelle.
            """
            df = df.reset_index()

            # Numerische Spalten verwenden
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    pass

            numeric_cols = df.select_dtypes(include=['number']).columns
            
            typst_table = df.style.format(
                precision=3,
                subset=numeric_cols,
                decimal=",",
                thousands="_"
            ).hide(axis="index").to_typst().replace("$", "")

            # Spaltenbreite festlegen
            num_cols = len(df.columns) 
            typst_table = typst_table.replace("[], ", "")

            if cv_or_is == "is" or cv_or_is == "cv":
                auto_align_column_string = f"columns: (auto, auto, auto, 3.7cm, auto, auto, 3.7cm, 3.7cm, 3.7cm),"
            elif not cv_or_is:
                auto_align_column_string = f"columns: ({', '.join(['auto'] * num_cols)}),"
            # Spalte linksbündig, die andere rechtsbündig
            table_align_string = "  align: (col, row) => if col == 0 { left } else { right },"
    
            lines = typst_table.split("\n")
    
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
            ).replace("[/]", "[]").replace("000]", "]") # Workaround: Typst-Export erzeugt bei manchen gerundeten Werten Endungen wie "000]", die hier entfernt werden
    
            return typst_table.replace("group", "Gruppe").replace("chapter", "Kapitel").replace("Group", "Gruppe").replace("Chapter", "Kapitel").replace("Base)", "Baseline)")

    def generate_regression_coefficients_appendix(self):
        """
        Erstellt eine Übersicht der Regressionskoeffizienten für den Anhang.
        """
        output_vars = ["entl_grd", "typ_vwd"]
        subset = "gesamt"
        input_var = "codes"
        sdx = True
        kovarianten = True
        icd_structures = ["group", "chapter"]

        auswertungsstring = f"""
        #show table.cell.where(y: 0): strong
        #page(flipped: true)[

        == Regressionskoeffizienten        
        """
        prev_output_var = None
        for output_var in output_vars:
            for icd_structure in icd_structures:
                path = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={output_var}", f"subset={subset}", f"input_var={input_var}", f"sdx={sdx}", f"{icd_structure}_kovarianten={kovarianten}.html")

                if output_var != prev_output_var:
                    if output_var == "entl_grd":
                        auswertungsstring += """
                        
                        === Mortalität
                        """
                    else:
                        auswertungsstring += """
                        
                        === Verweildauer
                        """

                if icd_structure == "group":
                    auswertungsstring += """
                    
                    ==== Gruppen-Ebene
                    """
                else:
                    auswertungsstring += """
                                        
                    ==== Kapitel-Ebene
                    """

                df = self._convert_html_table_to_dataframe(path, output_var)

                auswertungsstring += f"""
                {self._to_typst_formatted(df, cv_or_is=None)}
                """

                prev_output_var = output_var

        auswertungsstring += "]"

        auswertungsstring.replace("[],", "")

        path = os.path.join(self.AUSARBEITUNG_DIR, "regressionskoeffizienten.typ")

        with open(path, "w", encoding="utf-8") as f:
            f.write(auswertungsstring.replace("[],", ""))

    def _merge_calc_deltas(self, dfs_mod, dfs_base, val_column = "mean"):
        """
        Lädt die Daten.
        Berechnet den Unterschied zwischen den Metriken.
        Gibt die Ergebnisse als strukturiertes DataFrame zurück.
        """
        # Daten extrahieren
        group_mod = dfs_mod["group"].rename(columns={"Unnamed: 0": "Metrik", val_column: "Gruppe (Modell)"})[["Metrik", "Gruppe (Modell)"]].set_index("Metrik").T
        chapter_mod = dfs_mod["chapter"].rename(columns={"Unnamed: 0": "Metrik", val_column: "Kapitel (Modell)"})[["Metrik", "Kapitel (Modell)"]].set_index("Metrik").T
        group_base = dfs_base["group"].rename(columns={"Unnamed: 0": "Metrik", val_column: "Gruppe (Baseline)"})[["Metrik", "Gruppe (Baseline)"]].set_index("Metrik").T
        chapter_base = dfs_base["chapter"].rename(columns={"Unnamed: 0": "Metrik", val_column: "Kapitel (Baseline)"})[["Metrik", "Kapitel (Baseline)"]].set_index("Metrik").T

        # Dataframes zusammenführen
        df = pd.concat([group_mod, chapter_mod, group_base, chapter_base]).T
        df = df.apply(pd.to_numeric, errors='coerce')

        # Deltas berechnen
        # Vier Vergleichsachsen: Granularität (Gruppe vs Kapitel) und Modelltyp (Modell vs Baseline), jeweils gekreuzt
        df["Δ Group - Chapter #linebreak() (Modell)"] = df["Gruppe (Modell)"] - df["Kapitel (Modell)"]
        df["Δ Group - Chapter #linebreak() (Baseline)"] = df["Gruppe (Baseline)"] - df["Kapitel (Baseline)"]
        df["Δ Gruppe #linebreak() (Modell - Base)"] = df["Gruppe (Modell)"] - df["Gruppe (Baseline)"]
        df["Δ Kapitel #linebreak() (Modell - Base)"] = df["Kapitel (Modell)"] - df["Kapitel (Baseline)"]

        # Konfidenzintervalle extrahieren
        has_ci = "auroc_ci_lower" in df.index and "auroc_ci_upper" in df.index
        if has_ci:
            ci_values = {
                "Gruppe (Modell)": f"[{df.loc['auroc_ci_lower', 'Gruppe (Modell)']:.3f}; {df.loc['auroc_ci_upper', 'Gruppe (Modell)']:.3f}]".replace(".", ","),
                "Kapitel (Modell)": f"[{df.loc['auroc_ci_lower', 'Kapitel (Modell)']:.3f}; {df.loc['auroc_ci_upper', 'Kapitel (Modell)']:.3f}]".replace(".", ","),
                "Gruppe (Baseline)": f"[{df.loc['auroc_ci_lower', 'Gruppe (Baseline)']:.3f}; {df.loc['auroc_ci_upper', 'Gruppe (Baseline)']:.3f}]".replace(".", ","),
                "Kapitel (Baseline)": f"[{df.loc['auroc_ci_lower', 'Kapitel (Baseline)']:.3f}; {df.loc['auroc_ci_upper', 'Kapitel (Baseline)']:.3f}]".replace(".", ","),
                "Δ Group - Chapter #linebreak() (Modell)": "/",
                "Δ Group - Chapter #linebreak() (Baseline)": "/",
                "Δ Gruppe #linebreak() (Modell - Base)": "/",
                "Δ Kapitel #linebreak() (Modell - Base)": "/",
            }
            df = df.drop(index=["auroc_ci_lower", "auroc_ci_upper"], errors="ignore")

        desired_columns = [
            "Gruppe (Modell)", "Kapitel (Modell)", "Δ Group - Chapter #linebreak() (Modell)",
            "Gruppe (Baseline)", "Kapitel (Baseline)", "Δ Group - Chapter #linebreak() (Baseline)",
            "Δ Gruppe #linebreak() (Modell - Base)", "Δ Kapitel #linebreak() (Modell - Base)"
        ]
        df = df[desired_columns]

        df = df.astype(object)

        # Konfidenzintervalle hinzufügen
        if has_ci:
                df.loc["auroc_ci", :] = ""
                for col, text_interval in ci_values.items():
                    df.loc["auroc_ci", col] = text_interval
        
        return df

    def _scalar_cv_dfs_to_typst_wrapper(self, dfs_mod, dfs_base, output_var):
        """
        Wrapper Funktion zum Berechnen der Unterschiede und Umwandeln in Typst Code.
        Verwendet die skalaren Metriken aus der Kreuzvalidierung.
        """
        df = self._merge_calc_deltas(dfs_mod, dfs_base, val_column="mean")

        brier_or_rps = "BS" if output_var == "entl_grd" else "RPS"
        bss_or_rpss = "BSS" if output_var == "entl_grd" else "RPSS"

        df = df.rename(index={
                "overfit_gap": "Overfitting Gap", 
                "auroc": "AUROC", 
                "auroc_ci": "AUROC KI", 
                "rps_or_brier": brier_or_rps, 
                "rpss_or_bss": bss_or_rpss, 
            })

        return self._to_typst_formatted(df.sort_index(), cv_or_is="cv")

    def _scalar_is_dfs_to_typst_wrapper(self, dfs_mod, dfs_base):
        """
        Wrapper Funktion zum Berechnen der Unterschiede und Umwandeln in Typst Code.
        Verwendet die skalaren Metriken vom Evaluieren auf dem gesamten Datensatz.
        """
        df = self._merge_calc_deltas(dfs_mod, dfs_base, val_column="Wert")
        df = df.rename(index={"bic": "BIC"})
        return self._to_typst_formatted(df, cv_or_is="is")

    def _plot_roc_combined(self, output_var):
        """
        Erstellt eine Visualisierung der ROC-Kurven mit Vergleich zwischen den Granularitätsstufen und mit der Baseline.
        """
        if self.subset != "gesamt":
            key = list(self.subset.keys())[0]
            values = ", ".join(str(v) for v in self.subset.values())
            subset_str = f"subset {key}={values.replace("'", "").replace("[", "").replace("]", "")}"
        else:
            subset_str = "subset=gesamt"
    
        if self.kovarianten:
            if isinstance(self.subset, dict) and "aufn_anl" in self.subset.keys() and len(self.subset["aufn_anl"]) <= 1:
                kov = "['typ_alter', 'sex', 'anzahl_nebendiagnosen']"
            else:
                kov = "['typ_alter', 'sex', 'aufn_anl', 'anzahl_nebendiagnosen']"
        else:
            kov = None

        # Pfade definieren
        path_mod = os.path.join(SCORES_DIR, "scores_postprocessed", f"output_var={output_var}", subset_str, f"input_var={self.input_var}", f"sdx={self.use_sdx}", f"kovarianten={kov}")
        path_base = os.path.join(SCORES_DIR, "scores_postprocessed", f"output_var={output_var}", "subset=gesamt", f"input_var=codes", "sdx=True", f"kovarianten=['typ_alter', 'sex', 'aufn_anl', 'anzahl_nebendiagnosen']")

        # DataFrames laden
        roc_group_mod = pd.read_csv(os.path.join(path_mod, "cross_validation", "group_roc_aggregated.csv"), index_col=0)
        roc_chapter_mod = pd.read_csv(os.path.join(path_mod, "cross_validation", "chapter_roc_aggregated.csv"), index_col=0)
        roc_group_base = pd.read_csv(os.path.join(path_base, "cross_validation", "group_roc_aggregated.csv"), index_col=0)
        roc_chapter_base = pd.read_csv(os.path.join(path_base, "cross_validation", "chapter_roc_aggregated.csv"), index_col=0)

        # ROC Kurven plotten
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(roc_group_mod["fpr"], roc_group_mod["tpr"], label="Gruppe (Modell)", color="blue", linestyle="-")
        ax.plot(roc_chapter_mod["fpr"], roc_chapter_mod["tpr"], label="Kapitel (Modell)", color="orange", linestyle="-")
        ax.plot(roc_group_base["fpr"], roc_group_base["tpr"], label="Gruppe (Baseline)", color="blue", linestyle="--")
        ax.plot(roc_chapter_base["fpr"], roc_chapter_base["tpr"], label="Kapitel (Baseline)", color="orange", linestyle="--")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Zufallsklassifikator")

        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend()
        ax.set_title("ROC Kurven im Direktvergleich")
        ax.grid(True)

        path = os.path.join(path_mod, "cross_validation", "roc_curves_aggregated.png")
        fig.savefig(path, dpi=300)
        plt.close()

        path = path.replace(str(PROJECT_ROOT), "").replace("\\", "/").replace(r"\\", "")
        return f'#image("{path}")'

    def _plot_calibration_belt(self, output_var, use_sdx):
        """
        Erstellt eine Visualisierung des Kalibrierungsgürtels.
        """
        if output_var == "typ_vwd":
            return ""

        # Subgruppentext ändern
        if self.subset != "gesamt":
            key = list(self.subset.keys())[0]
            values = list(self.subset.values())[0]
            subset_str = f"subset {key}={values}"
        else:
            subset_str = "subset=gesamt"

        # Kovariantentext ändern
        if self.kovarianten:
            if isinstance(self.subset, dict) and "aufn_anl" in self.subset.keys() and len(self.subset["aufn_anl"].split(",")) <= 1:
                kov = "['typ_alter', 'sex', 'anzahl_nebendiagnosen']"
            else:
                kov = "['typ_alter', 'sex', 'aufn_anl', 'anzahl_nebendiagnosen']"
        else:
            kov = None

        ergebnis_str = ""
        path_to_scores_dir = os.path.join(SCORES_DIR, "scores_postprocessed", f"output_var={output_var}", subset_str, f"input_var={self.input_var}", f"sdx={use_sdx}", f"kovarianten={kov}")

        for icd_structure in self.icd_structures:
            path = os.path.join(path_to_scores_dir, "cross_validation", f"{icd_structure}_calibration_belt_aggregated.png")
            path = path.replace(str(PROJECT_ROOT), "").replace("\\", "/").replace(r"\\", "")
            ergebnis_str += f'{icd_structure.replace("group", "ICD-Struktur: Gruppe").replace("chapter", "ICD-Struktur: Kapitel")}:\n#image("{path}", width: 90%)\n'
            ergebnis_str += "#pagebreak() "
        return ergebnis_str

    def generate_evaluation_reports(self, subset, kovarianten, input_var, use_sdx):
        """
        Erstellt Typst-Auswertungsdokumente für eine Modellkonfiguration.
        Vergleicht Gruppe vs. Kapitel und Modell vs. Baseline für Mortalität und Verweildauer.

        Args:
            subset: Subgruppenfilter, z.B. {'aufn_anl': ['N']}, oder 'gesamt'
            kovarianten: Liste der Kovariaten oder None
            input_var: Art der Eingabevariablen ('codes', 'charlson', 'elixhauser')
            use_sdx: Ob Nebendiagnosen verwendet werden
        """
        self.subset = subset
        self.kovarianten = kovarianten
        self.input_var = input_var
        self.use_sdx = use_sdx

        scalar_scores_cv = {}
        scalar_scores_is = {}

        for output_var in self.output_vars:
            # Dictionaries anlegen
            dfs_mod_cv, dfs_base_cv = {}, {}
            dfs_mod_is, dfs_base_is = {}, {}
            if output_var != "typ_vwd":
                dfs_mod_calbelt, dfs_base_calbelt = {}, {}

            if subset == "gesamt":
                subset_str = "subset=gesamt"
            else:
                keys = list(subset.keys())[0]
                values = list(subset.values())[0]
                subset_str = f"subset {keys}={values}".replace("[", "").replace("]", "").replace("'", "")

            for icd_structure in self.icd_structures:
                # Daten einlesen
                path_mod = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={output_var}", f"{subset_str}", f"input_var={input_var}", f"sdx={self.use_sdx}", f"{icd_structure}_kovarianten={kovarianten}.html")
                path_base = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={output_var}", f"subset=gesamt", f"input_var=codes", "sdx=True", f"{icd_structure}_kovarianten=True.html")
    
                dfs_mod_cv[icd_structure] = pd.read_html(path_mod)[0].round(3)
                dfs_base_cv[icd_structure] = pd.read_html(path_base)[0].round(3)
                dfs_mod_is[icd_structure] = pd.read_html(path_mod)[1].round(0)
                dfs_base_is[icd_structure] = pd.read_html(path_base)[1].round(0)

                if output_var != "typ_vwd":
                    dfs_mod_calbelt[icd_structure] = pd.read_html(path_mod)[2]
                    dfs_base_calbelt[icd_structure] = pd.read_html(path_base)[2]

            # Skalare Werte transformieren
            scalar_scores_cv[output_var] = self._scalar_cv_dfs_to_typst_wrapper(dfs_mod_cv, dfs_base_cv, output_var)
            scalar_scores_is[output_var] = self._scalar_is_dfs_to_typst_wrapper(dfs_mod_is, dfs_base_is).replace(",]", "]")

        auswertungsstring = f"""
            #show regex("(^|[^0])([0])\.([0-9]+)"): it => {{
            it.text.replace(".", ",")
        }}
        #show regex("\d+(_\d{{3}})+\\b"): it => {{
            it.text.replace("_", ".")
        }}
        #page(flipped: true)[

        #show table.cell.where(y: 0): strong

        = Auswertung
        
        Hyperparameter:

        Subgruppe: {subset_str.replace("subset ", "").replace("N", "Notfall").replace("M", "medizinischer Partition zugeteilt").replace("O", "operativer Partition zugeteilt").replace("aufn_anl=", "Aufnahmeanlass ist ").replace("E,V,G", "kein Notfall").replace("partition=", "Krankenhausfall wird  ")},
        unabhängige Variable: {input_var.replace("codes", "Codes").replace("charlson", "Charlson").replace("elixhauser", "Elixhauser")},
        Verwendung Kovarianten: {str(kovarianten).replace("True", "Ja").replace("False", "Nein")},
        Verwendung Nebendiagnosen: {str(use_sdx).replace("True", "Ja").replace("False", "Nein")}

        === Mortalität

        ==== Evaluiert auf gesamten Datensatz
        {scalar_scores_is["entl_grd"]}


        ==== Kreuzvalidiert
        {scalar_scores_cv["entl_grd"]}

        #pagebreak()

        ===== ROC-Kurve
        {self._plot_roc_combined("entl_grd")}

        #pagebreak()

        ===== Kalibrierungsgürtel
        {self._plot_calibration_belt("entl_grd", use_sdx)}

        === Verweildauer
        ==== Evaluiert auf gesamten Datensatz
                {scalar_scores_is["typ_vwd"]}

        ==== Kreuzvalidiert
        {scalar_scores_cv["typ_vwd"]}

        #pagebreak()

        ===== ROC-Kurve
        {self._plot_roc_combined("typ_vwd")}
    
        

        ]
        """

        # Dateiname kodiert alle vom Baseline-Setting abweichenden Hyperparameter; wenn keiner abweicht, wird "baseline" verwendet
        änderungsstring = ""
        
        is_base = True
        if subset == "gesamt":
            pass
        else:
            if isinstance(subset, dict):
                k = list(subset.keys())[0]
                v = list(subset.values())[0]
                subset_filename_part = f"subset_{k}_{v}"
            else:
                subset_filename_part = f"subset_{subset}"
            änderungsstring += subset_filename_part
            is_base = False
        if input_var == "codes":
            pass
        else:
            änderungsstring += f"input_var = {input_var}"
            is_base = False
        if use_sdx == True:
            pass
        else:
            änderungsstring += f"sdx = {use_sdx}"
            is_base = False
        if kovarianten == True:
            pass
        else:
            änderungsstring += f"kovarianten = {kovarianten}"
            is_base = False
    
        prefix = "auswertung_baseline" if is_base else "auswertung_"

        dateiname = (
            f"{prefix}{änderungsstring}.typ"
        )
        os.makedirs(os.path.join(self.AUSARBEITUNG_DIR, "auswertungen"), exist_ok=True)

        # Datei speichern
        ziel_pfad = os.path.join(self.AUSARBEITUNG_DIR, "auswertungen", dateiname)
        with open(ziel_pfad, "w", encoding="utf-8") as f:
            f.write(auswertungsstring)