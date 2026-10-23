"""
Nachverarbeitung der Modellmetriken und Erstellung von HTML-Berichten mit skalaren Kennzahlen, ROC-Kurve, Kalibrierungsgürtel und statistischer Analyse.
"""

from drg_risk import SCORES_DIR, REPORT_DIR
from drg_risk.utils.logger import Logger
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

class Score_PostProcessor():
    """
    Klasse zum Verarbeiten der berechneten Metriken.
    Erstellt einen Bericht zur Aufbereitung der Metriken.
    """
    def __init__(self, output_var, use_sdx, icd_structure, subset, n_folds, input_var, kovarianten):
        """
        Args:
            output_var: Zielvariable ('entl_grd' für Mortalität, 'typ_vwd' für Verweildauer)
            use_sdx: Ob Nebendiagnosen verwendet werden
            icd_structure: 'group' oder 'chapter'
            subset: Optionaler Filter, z.B. {'aufn_anl': ['N']}
            n_folds: Anzahl der Kreuzvalidierungs-Folds
            input_var: Art der Eingabevariablen ('codes', 'charlson', 'elixhauser')
            kovarianten: Ob demografische Kovariaten verwendet werden
        """
        self.output_var =           output_var
        self.use_sdx =              use_sdx
        self.icd_structure =        icd_structure
        self.subset =               subset
        self.kovarianten =          kovarianten
        self.input_var =            input_var
        self.n_folds =              n_folds
        self.logger =               Logger(name = "score postprocessor")

        # Wenn Subgruppen verwendet wurden
        if self.subset:
            key = list(self.subset.keys())[0]
            value = list(self.subset.values())[0]
            value_str = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            self.subset_str = f"subset {key}={value_str}"
        else:
            self.subset_str = "subset=gesamt"

        # Pfad definieren
        self.path_to_scores_dir = os.path.join(
            SCORES_DIR,
            "scores_postprocessed",
            f"output_var={self.output_var}",
            self.subset_str,
            f"input_var={self.input_var}",
            f"sdx={self.use_sdx}",
            f"kovarianten={self.kovarianten}"
        )

    def _postprocess_scalar_scores_cross_validated(self):
        """
        Verarbeitet die skalaren Metriken der Kreuzvalidierung.
        """

        # Metriken laden
        scalar_scores_cv = []
        for fold in range(1, self.n_folds + 1):
            with open(os.path.join(self.path_to_scores_dir, "cross_validation", f"{self.icd_structure}_scalar_scores_fold_{fold}.json"), "r") as f:
                scalar_scores_cv.append(json.load(f))

        scores = {}
        keys = scalar_scores_cv[0].keys()

        for key in keys:
            scores[key] = [score_cv[key] for score_cv in scalar_scores_cv]

        # Metriken aufbereiten und in einem DataFrame zusammenfassen
        data = {}
        for key in scores.keys():
            if any(v is None for v in scores[key]):
                continue
            # area_under_the_curve ist als Dict mit AUROC + Konfidenzintervall gespeichert (anders als restliche Metriken), daher gesonderte Extraktion
            if key == "area_under_the_curve":
                list_auroc = []
                list_ci_lower = []
                list_ci_upper = []

                for values in scores[key]:
                    list_auroc.append(float(values["AUROC"]))
                    list_ci_lower.append(float(values["CI_lower"]))
                    list_ci_upper.append(float(values["CI_upper"]))
                data["auroc"] = {"mean": np.mean(list_auroc), "std": np.std(list_auroc)}
                data["auroc_ci_lower"] = {"mean": np.mean(list_ci_lower), "std": np.std(list_ci_lower)}
                data["auroc_ci_upper"] = {"mean": np.mean(list_ci_upper), "std": np.std(list_ci_upper)}
            else:
                data[key] = {"mean": np.mean(scores[key]), "std": np.std(scores[key])}

        return pd.DataFrame(data.values(), index=data.keys())
    
    def _plot_roc_aggregated(self):
        """
        Funktion zum Zeichnen der ROC-Kurve
        """

        # Daten laden
        roc_aggregated = pd.read_csv(os.path.join(self.path_to_scores_dir, "cross_validation", f"{self.icd_structure}_roc_aggregated.csv"), index_col=0)

        # ROC-Kurve zeichnen
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(roc_aggregated["fpr"], roc_aggregated["tpr"], label = "ROC-Kurve")
        ax.plot([0, 1], [0, 1], color = "grey", linestyle = "--", label = "Zufallsklassifikator")

        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")

        ax.legend()
        ax.set_title("ROC Kurve")
        ax.grid()

        # Bild speichern
        fig.savefig(os.path.join(self.path_to_scores_dir, "cross_validation", f"{self.icd_structure}_roc_aggregated.png"), dpi = 300)
        
        return fig, ax
    
    def _postprocess_scalar_scores_insampled(self):
        """
        Skalare Metriken auf dem gesamten Datensatz laden.
        """
        with open(os.path.join(self.path_to_scores_dir, "insample", f"{self.icd_structure}_scalar_scores.json"), "r") as f:
            scalar_scores_insampled = json.load(f)

        return scalar_scores_insampled
    
    def _postprocess_stats_analysis(self):
        """
        Lädt die statistische Analyse (Koeffizienten, p-Werte, Odds Ratios) aus der In-Sample-Evaluierung.
        """

        # Metriken laden
        stat_analysis_insampled = pd.read_csv(os.path.join(self.path_to_scores_dir, "insample", f"{self.icd_structure}_stat_analysis_scores.csv"), index_col=0)

        return stat_analysis_insampled
    
    def generate_score_report(self, start_time, info_string):
        """
        Erstellt einen HTML-Bericht mit allen Metriken einer Modellkonfiguration.

        Args:
            start_time: Startzeitpunkt des Trainings (pd.Timestamp) für Laufzeitberechnung
            info_string: HTML-String mit Modellkonfigurationsdetails
        """
        self.logger.info(f"Erstelle Ordner in {REPORT_DIR}")
        
        path = os.path.join(REPORT_DIR, "scores_postprocessed", f"output_var={self.output_var}", self.subset_str, f"input_var={self.input_var}", f"sdx={self.use_sdx}")
        os.makedirs(path, exist_ok=True)
        kovarianten = bool(self.kovarianten)

        path = os.path.join(path, f"{self.icd_structure}_kovarianten={kovarianten}.html")

        self.logger.info(f"Erstelle report in {path}")
        
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(
                f"<h1>Score Report: {self.output_var} ("
                f"SDX: {self.use_sdx}, "
                f"ICD-Structure: {self.icd_structure}, "
                f"Input Variable: {self.input_var}, "
                f"Subset: {self.subset_str.split("subset")[1:]}, "
                f"Kovarianten: {kovarianten}"
                f")</h1>"
            )

            f.write(info_string)
            
            self.logger.info(f"Schreibe skalare Werte")
            f.write("<h2>Scalar Scores</h2>")
            f.write("<h3>Cross Validated</h3>")
            f.write(self._postprocess_scalar_scores_cross_validated().to_html(float_format=lambda x: f"{x:.4f}"))
            f.write("<h3>Insampled</h3>")
            f.write(pd.DataFrame.from_dict(self._postprocess_scalar_scores_insampled(), orient="index", columns = ["Wert"]).to_html(float_format=lambda x: f"{x:.4f}"))

            self.logger.info(f"Schreibe ROC Kurve")
            fig_roc, _ = self._plot_roc_aggregated()

            if fig_roc is not None:
                f.write("<h2>ROC Aggregated</h2>")
                buf = io.BytesIO()
                fig_roc.savefig(buf, format="png", bbox_inches="tight")
                plt.close(fig_roc)
                img_64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                f.write(
                    f'<img src="data:image/png;base64,{img_64}" width="450" style="margin-right:20px;">'
                )
            
            self.logger.info(f"Schreibe Calibration Belt")
            f.write("<h2>Calibration Belt</h2>")
            try:

                path = os.path.join(self.path_to_scores_dir, "cross_validation", f"{self.icd_structure}_calibration_belt_aggregated.png")
                with open(path, "rb") as img_f:
                    img_64 = base64.b64encode(img_f.read()).decode("utf-8")
                f.write(
                    f'<img src="data:image/png;base64,{img_64}" width="450" style="margin-right:20px;">'
                )
            except FileNotFoundError:
                # FileNotFoundError erwartet, wenn für diese Konfiguration kein Calibration Belt berechnet wurde
                f.write("Keine Daten für Calibration Belt.")
                self.logger.info("Keine Daten für Calibration Belt.")

            self.logger.info(f"Schreibe statistische Analyse")
            f.write("<h2>Stats Analysis</h2>")
            df = self._postprocess_stats_analysis()

            # Nur signifikante Variablen behalten: Odds-Ratio-KI darf die 1 nicht inkludieren
            df = df[~((df["odds_CI_lower"] < 1.) & (df["odds_CI_upper"] > 1.))]
 

            f.write(df.to_html(float_format=lambda x: f"{x:.4f}"))

            f.write("Gesamtlaufzeit: " + str((pd.Timestamp.now() - start_time).total_seconds()))

            self.logger.info(f"Statistische Analyse geschrieben.")

            f.flush()