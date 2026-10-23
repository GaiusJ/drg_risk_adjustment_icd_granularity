"""
Evaluation der Modellgüte für Mortalitäts- und Verweildauermodelle.
Berechnet skalare Metriken, statistische Kennzahlen und nicht-skalare Metriken (ROC, Kalibrierungsgürtel) und speichert sie als JSON/CSV/PNG.
"""

from drg_risk.evaluation.scores import (
    overfit_gap,
    area_under_the_roc_curve,
    bic,
    brier_score,
    calibration_belt,
    roc,
    brier_skill_score,
    ranked_probability_skill_score,
    ranked_probability_score,
)
from drg_risk import SCORES_DIR
from drg_risk.utils.logger import Logger

from scipy.stats import chi2
from scipy.sparse import hstack, csr_matrix
import os
import json
import numpy as np
import pandas as pd


class Evaluator():
    """
    Klasse zum Evaluieren der Modellgüte.
    Evaluiert das Modell und speichert die Metriken.
    """
    def __init__(self, output_var, use_sdx, feature_cols, icd_structure, subset, input_var, kovarianten):
        """
        Args:
            output_var: Zielvariable ('entl_grd' für Mortalität, 'typ_vwd' für Verweildauer)
            use_sdx: Ob Nebendiagnosen verwendet werden
            feature_cols: Liste der Feature-Spaltennamen
            icd_structure: 'group' oder 'chapter'
            subset: Optionaler Filter, z.B. {'aufn_anl': ['N']}
            input_var: Art der Eingabevariablen ('codes', 'charlson', 'elixhauser')
            kovarianten: Ob demografische Kovariaten verwendet werden
        """
        self.output_var =           output_var
        self.use_sdx =              use_sdx
        self.logger =               Logger(name = "evaluator")
        self.feature_cols =         feature_cols
        self.icd_structure =        icd_structure
        self.subset =               subset
        self.input_var =            input_var
        self.kovarianten =          kovarianten

        # wenn Subgruppen verwendet werden
        if self.subset:
            key = list(self.subset.keys())[0]
            value = list(self.subset.values())[0]
            value_str = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            self.subset_str = f"subset {key}={value_str}"
        else:
            self.subset_str = "subset=gesamt"

        self.path_to_scores_dir = os.path.join(
            SCORES_DIR,
            "scores_postprocessed",
            f"output_var={self.output_var}",
            self.subset_str,
            f"input_var={self.input_var}",
            f"sdx={self.use_sdx}",
            f"kovarianten={self.kovarianten}"
        )

    def evaluate_model_in_sample(self, y_train, y_train_pred_proba, num_params, model, X_train):
        """
        Evaluiert das Modell auf dem gesamten Trainingsdatensatz (In-Sample).
        Berechnet BIC und statistische Kennzahlen (Koeffizienten, p-Werte, Odds Ratios).

        Returns:
            Dict mit 'scalar_scores' und 'stat_analysis_scores'
        """
        scalar_scores = {
            "bic":  bic(y_train, y_train_pred_proba, num_params=num_params),    # BIC wird nur einmal auf dem vollen Trainingsdatensatz berechnet, nicht pro CV-Fold
        }

        # Berechnet p-Werte, Konfidenzintervalle, Odds Ratios
        stat_analysis_scores = self._statistical_analysis(feature_names=self.feature_cols, X=X_train, model = model, y=y_train)

        return {
            "scalar_scores":            scalar_scores,
            "stat_analysis_scores":      stat_analysis_scores,
        }

    def save_in_sample_scores(self, scores):
        """
        Speichert Metriken berechnet auf dem gesamten Datensatz.
        """

        # Pfad definieren
        path_to_scores = os.path.join(self.path_to_scores_dir, "insample")
        self.logger.info(f"Speichere Scores in {path_to_scores}")
        os.makedirs(path_to_scores, exist_ok=True)

        # Skalare Werte speichern
        with open(os.path.join(path_to_scores, f"{self.icd_structure}_scalar_scores.json"), "w") as f:
            json.dump(scores["scalar_scores"], f, indent = 4)

        # Statistische Kennzahlen speichern
        scores["stat_analysis_scores"].to_csv(os.path.join(path_to_scores, f"{self.icd_structure}_stat_analysis_scores.csv"))

        self.logger.info("Scores erfolgreich gespeichert.")

    def evaluate_model_out_sample(self, fold, y_train, y_train_pred_proba, y_test, y_test_pred_proba, y_test_pred, X_test, feature_cols, model):
        """
        Evaluiert das Modell auf dem Testdatensatz eines Kreuzvalidierungs-Folds.
        Für Mortalität: Brier Score und BSS; für Verweildauer: RPS und RPSS, da letztere die ordinale Struktur der Verweildauerkategorien berücksichtigen.
        """
        self.logger.info(f"Evaluiere Modell: Fold {fold + 1}")

        # Skalare Werte berechnen
        scalar_scores = {
            "overfit_gap":                          overfit_gap(y_train, y_train_pred_proba, y_test, y_test_pred_proba),
            "area_under_the_curve":                 area_under_the_roc_curve(y_test, y_test_pred_proba),
            "rps_or_brier":                         brier_score(y_test, y_test_pred_proba) if self.output_var == "entl_grd" else ranked_probability_score(y_test, y_test_pred_proba),
            "rpss_or_bss":                          brier_skill_score(y_test, y_test_pred_proba)if self.output_var == "entl_grd" else ranked_probability_skill_score(y_test, y_test_pred_proba),
        }

        self.logger.info(f"Model evaluiert")
        return {
            "scalar_scores":            scalar_scores,
        }
    
    def evaluate_model_out_sample_aggregated(self, y_test_fold_combined, y_test_fold_combined_pred_probas):
        """
        Funktion zum Berechnen der nicht skalaren Metriken während der Kreuzvalidierung.
        """

        # Fügt alle Testdatensätze über die gesamte Kreuzvalidierung zusammen (Pooling)
        y_test_true = np.concatenate(y_test_fold_combined)
        y_test_pred_proba = np.concatenate(y_test_fold_combined_pred_probas)

        # Berechnet Kalibierungsgürtel und ROC Kurve für gesamten Datensatz
        return {
            "calibration_belt":                     calibration_belt(y_test_true,y_test_pred_proba),
            "roc":                                  roc(y_test_true, y_test_pred_proba)
        }
    
    def save_aggregated_scores_out_sample(self, scores):
        """
        Speichert aggregierte Metriken berechnet während der Kreuzvalidierung.
        """

        # Pfad definieren
        scores = {k: v for k, v in scores.items() if v is not None}
        path_to_scores = os.path.join(self.path_to_scores_dir, "cross_validation")
        self.logger.info(f"Speichere Scores in {path_to_scores}")
        os.makedirs(path_to_scores, exist_ok=True)

        # Kalibrierungsgürtel
        if "calibration_belt" in scores.keys():
            scores["calibration_belt"][1].savefig(os.path.join(path_to_scores, f"{self.icd_structure}_calibration_belt_aggregated.png"), dpi = 300)

        # ROC Kurve
        if "roc" in scores.keys():
            df = pd.DataFrame({"fpr": scores["roc"][0], "tpr": scores["roc"][1]})
            df.to_csv(os.path.join(path_to_scores, f"{self.icd_structure}_roc_aggregated.csv"))

        self.logger.info("Scores erfolgreich gespeichert.")

    def _statistical_analysis(self, feature_names, X, model, y):
        """
        Berechnet Koeffizienten, Standardfehler, p-Werte (Wald-Test) und Odds Ratios mit 95%-Konfidenzintervallen.
        Für Mortalität: Fisher-Information aus der logistischen Likelihood.
        Für Verweildauer: Standardfehler via Stichprobenannäherung.

        Args:
            feature_names: Liste der Feature-Namen
            X: Designmatrix
            model: Trainiertes Modell
            y: Zielvariable
        Returns:
            DataFrame mit Koeffizienten, OR, SE, p-Wert und KI
        """
        self.logger.info("Berechne statistische Kennzahlen.")

        # Koeffizienten extrahieren
        if self.output_var == "entl_grd":
            feature_names = ["intercepts"] + feature_names
            coef = np.concatenate([model.intercept_, model.coef_.flatten()])
            pred_proba = model.predict_proba(X)[:, 1]
            V = pred_proba * (1 - pred_proba)
        elif self.output_var == "typ_vwd":
            coef = np.concatenate([model.coefs, model.thresholds])
            feature_names = feature_names + [f"threshold_{k}" for k in range(len(model.thresholds))]

        # Odds Ratios berechnen
        odds_ratio = np.exp(coef)

        # Standardabweichung berechnen
        if self.output_var == "entl_grd":
            # standard error: https://mybiostats.wordpress.com/wp-content/uploads/2015/03/the_multiple_logistic_regression_model.pdf
            intercept = csr_matrix(np.ones((X.shape[0], 1), dtype=np.float32))
            X_aug = hstack([intercept, X], format="csr")

            # Fisher-Matrix
            VX = X_aug.T.multiply(V[None, :])
            I = VX.dot(X_aug).toarray()
            # Kleine Regularisierung gegen numerische Instabilität/Singularität bei der Matrixinversion
            I += np.eye(I.shape[0]) * 1e-5

            var = np.linalg.inv(I)
            std = np.sqrt(np.diag(var))
        
        elif self.output_var == "typ_vwd":
            # Standardabweichung, wurde nur auf einer stichprobe berechnet, da n=1,7 mio ist es jedoch vertretbar
            std = model.compute_std_error(X, y)
            

        # Konfidenzintervalle berechnen
        # Clipping verhindert Overflow bei np.exp() für Odds Ratios bei extrem großen/kleinen Koeffizienten
        coef_ci_lower = np.clip(coef - 1.96 * std, -500, 500)
        coef_ci_upper = np.clip(coef + 1.96 * std, -500, 500)

        odds_ci_lower = np.exp(coef_ci_lower)
        odds_ci_upper = np.exp(coef_ci_upper)

        # p-Wert berechnen: Wald test
        z_2 = (coef**2) / (std**2)
        p_value = chi2.sf(z_2, df = 1)

        self.logger.info("Statistische Kennzahlen berechnet.")

        return pd.DataFrame({
            "name": feature_names,
            "coefficient": coef,
            "odds_ratio": odds_ratio,
            "std_error": std,
            "p_value": p_value,
            "coef_CI_lower": coef_ci_lower,
            "coef_CI_upper": coef_ci_upper,
            "odds_CI_lower": odds_ci_lower,
            "odds_CI_upper": odds_ci_upper,
        })
    
    # Speichere Scores von Cross Validation
    def save_out_sample_scores(self, fold, scores):
        """
        Speichert skalare Metriken berechnet während der Kreuzvalidierung.
        """
        scores = {k: v for k, v in scores.items() if v is not None}
        path_to_scores = os.path.join(self.path_to_scores_dir, "cross_validation")
        self.logger.info(f"Speichere Scores in {path_to_scores}")
        os.makedirs(path_to_scores, exist_ok=True)

        # Skalare Werte speichern
        if "scalar_scores" in scores.keys():
            with open(os.path.join(path_to_scores, f"{self.icd_structure}_scalar_scores_fold_{fold + 1}.json"), "w") as f:
                json.dump(scores["scalar_scores"], f, indent = 4)

        self.logger.info("Scores erfolgreich gespeichert.")