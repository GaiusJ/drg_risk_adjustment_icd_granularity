"""
Metriken zur Evaluation von Mortalitäts- (binär) und Verweildauermodellen (ordinal).
Diskriminierung: AUROC, ROC. Kalibrierung: Brier Score/BSS, RPS/RPSS, Kalibrierungsgürtel.
Modellkomplexität: BIC, Overfit Gap.
"""

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_curve,
)
from confidenceinterval import roc_auc_score
import numpy as np
from pycaleva import CalibrationEvaluator

from drg_risk.evaluation.mauc import fun_mauc_ovo_ordinal

## Diskriminierungsmaße

def area_under_the_roc_curve(y_test, y_pred_proba):
    """
    Berechnet die Area under the Receiver Operator Curve für binäre und ordinale Klassifikationsmodelle.
    """

    # binäres Modell: Mortalität
    if np.ndim(y_pred_proba) == 1:
        AUROC, (lower, upper) = roc_auc_score(y_test.astype(int),
                        y_pred_proba,
                        confidence_level=0.95)       
        return {
            "AUROC": float(AUROC),
            "CI_lower": float(lower),
            "CI_upper": float(upper),
        }

    # multiclass Modell: Verweildauer
    elif np.ndim(y_pred_proba) == 2:
        AUROC, mauc_std = fun_mauc_ovo_ordinal(y_pred_proba, y_test)

        # Berechnung der Konfidenzintervalle
        ci_lower = np.clip(AUROC - 1.96 * mauc_std, 0., 1.)
        ci_upper =  np.clip(AUROC + 1.96 * mauc_std, 0., 1.)
        return {
            "AUROC": float(AUROC),
            "CI_lower": float(ci_lower),
            "CI_upper": float(ci_upper),
        }
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")
    
def roc(y_test, y_pred_proba):
    """
    Funktion zum Berechnen der Punkte der Receiver Operator Curve.
    """
    if np.ndim(y_pred_proba) == 1:
        return roc_curve(y_test, y_pred_proba)
    elif np.ndim(y_pred_proba) == 2:
        # One-vs-One Ansatz
        all_fpr = np.linspace(0, 1., 1000)
        tprs = np.zeros_like(all_fpr)
        pair_counter = 0
        for i in range(y_pred_proba.shape[1]):
            for j in range(i + 1, y_pred_proba.shape[1]):
                mask = (y_test == i) | (y_test == j)
                y_bin = (y_test[mask] == i).astype(int)

                fpr_ij, tpr_ij, _ = roc_curve(y_bin, y_pred_proba[mask, i])
                fpr_ji, tpr_ji, _ = roc_curve(1 - y_bin, y_pred_proba[mask, j])

                # Auf raster interpolieren, für jedes Klassenpaar (i,j) beide Richtungen (i vs j und j vs i) berechnen und mitteln -> makro-gemittelte OvO-ROC 
                tprs += np.interp(all_fpr, fpr_ij, tpr_ij)
                tprs += np.interp(all_fpr, fpr_ji, tpr_ji)
                
                pair_counter += 2
        return all_fpr, tprs / pair_counter
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")

## Kalibrierungsmaße

def brier_score(y_test, y_pred_proba):
    """
    Berechnet den Brier Score zur Evaluierung eines Modells. Lediglich möglich bei binären Entscheidungsmodell (Mortalität).
    """
    if np.ndim(y_pred_proba) == 1:
        return float(brier_score_loss(y_true = y_test, y_proba = y_pred_proba))
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")

def calibration_belt(y_test, y_pred_proba):
    """
    Berechnet für den binären Fall die Punkte des Kalibrierungsgürtels.
    """
    if np.ndim(y_pred_proba) == 1:
        ce = CalibrationEvaluator(y_test, y_pred_proba, outsample=True, n_groups="auto")
        return ce.calbelt(plot=False), ce.calibration_plot()
    elif np.ndim(y_pred_proba) == 2:
        return None
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")

def ranked_probability_score(y_test, y_pred_proba):
    """
    Berechnet den Ranked Probability Score. Dieser ist die Erweiterung des Brier Scores für ordinale Entscheidungsmodelle.
    """

    # One-Hot-Kodierung
    y_test_onehot = np.zeros_like(y_pred_proba)
    y_test_onehot[np.arange(len(y_test)), y_test] = 1

    # Kumulative Summe berechnen
    y_test_sum = np.cumsum(y_test_onehot, axis = 1)
    y_pred_proba_sum = np.cumsum(y_pred_proba, axis = 1)

    num_classes = 7  # Anzahl ordinaler Verweildauerkategorien (typ_vwd 1-7), siehe Campus-File

    # Normierte quadratische Abweichung pro Beobachtung berechnen
    rps_per_row =  1/(num_classes - 1)*np.sum((y_pred_proba_sum - y_test_sum)**2, axis = 1)

    # Mittelwert bilden
    return np.mean(rps_per_row)

def brier_skill_score(y_test, y_pred_proba):
    """
    Berechnet den Brier Skill Score. Dieser setzt den Brier Score des Modells mit dem eines Nullmodells in Relation.
    Bei dem Nullmodell handelt es sich um die Vorhersage der Klassenwahrscheinlichkeiten.
    """
    if np.ndim(y_pred_proba) == 1:
        brier_full = brier_score(y_test, y_pred_proba)
        # Nullmodell: konstante Vorhersage = Basisrate (mittlere Mortalitätsrate) für alle Fälle
        brier_null = brier_score(y_test, np.full_like(y_pred_proba, np.mean(y_test)))
        return float(1. - brier_full / brier_null)
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")

def ranked_probability_skill_score(y_test, y_pred_proba):
    """
    Berechnet den Ranked Probability Skill Score. Dieser setzt den Ranked Probability Score des Modells mit dem eines Nullmodells in Relation.
    Bei dem Nullmodell handelt es sich um die Vorhersage der Klassenwahrscheinlichkeiten. Der Ranked Probability Score ist die Erweiterung des Brier Skill Scores für ordinale Entscheidungsmodelle.
    """
    if np.ndim(y_pred_proba) == 2:
        rps_full = ranked_probability_score(y_test, y_pred_proba)

        num_classes = y_pred_proba.shape[1]
        class_frequencies = np.array([np.mean(y_test == k) for k in range(num_classes)])

        rps_int_means = np.tile(class_frequencies, (len(y_test), 1))

        rps_null = ranked_probability_score(y_test, rps_int_means)

        return float(1.-rps_full/rps_null)
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")


# Maße zur Beurteilung von Prädiktgewinn und Modellkomplexität

def _log_likelihood(y_train, y_train_proba):
    """
    Berechnet die logarithmierte Übereinstimmung zweier Reihen mithilfe der Likelihood Funktion.
    """
    if np.ndim(y_train_proba) == 1 or np.ndim(y_train_proba) == 2:
        # Log-Loss von sklearn ist bereits über n gemittelt; hier zurückskaliert auf Summe für BIC-Berechnung
        sample_size = len(y_train)
        return -1 * log_loss(y_train, y_train_proba) * sample_size
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")
    
def bic(y_train, y_train_proba, num_params):
    """
    Berechnet das Bayes-Informationskriterium.
    """
    if np.ndim(y_train_proba) == 1 or np.ndim(y_train_proba) == 2:
        log_likelihood = _log_likelihood(y_train, y_train_proba)
        count_data = len(y_train)
        return float(-2 * log_likelihood + num_params * np.log(count_data))
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")

def overfit_gap(y_train, y_train_proba, y_test, y_test_proba):
    """
    Berechnet den Unterschied des Verlustes auf dem Evaluierungsdatensatz und auf dem Testdatensatz.
    Ein großer Unterschied spricht für eine Überanpassung des Modells auf den Datensatz.
    """
    if np.ndim(y_train_proba) == 1 or np.ndim(y_train_proba) == 2:
        log_loss_train = log_loss(y_train, y_train_proba)
        log_loss_test = log_loss(y_test, y_test_proba)
        return float(abs(log_loss_test - log_loss_train))
    else:
        raise ValueError("Ungültige Größe von y_pred_proba")