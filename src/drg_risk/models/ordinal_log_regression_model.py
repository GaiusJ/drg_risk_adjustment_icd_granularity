"""
Implementierung des Proportional-Odds-Modells (McCullagh) für ordinale Regression.
Gradient via numba-parallelisierter analytischer Ableitung; Optimierung via L-BFGS-B.
Verwendet für die Verweildauer-Klassifikation (7 ordinale Kategorien).
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from drg_risk.utils.logger import Logger

# Sigmoid Funktion
@njit
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# Ableitung der Sigmoid Funktion
@njit
def _sigmoid_derivative(x):
    return _sigmoid(x) * (1 - _sigmoid(x))

# gibt derzeitige Schwellenwerte zurück
# exp() erzwingt positive Abstände zwischen Schwellenwerten -> garantiert monoton steigende Thresholds
@njit
def _get_thresholds(deltas):
    thresholds = np.zeros(len(deltas))

    thresholds[0] = deltas[0]
    for i in range(1, len(deltas)):
        thresholds[i] = thresholds[i - 1] + np.exp(deltas[i])

    return thresholds

@njit
def _calculate_gradient(coefs, deltas, num_classes, X, y):
    """
    Berechnet den Gradienten des ordinalen Modells.
    """

    N, p = X.shape

    # Koeffizienten, Schwellenwerte und Abstand zwischen den Schwellenwerten initialisieren
    gradient_coef = np.zeros(p)
    gradient_delta = np.zeros(num_classes - 1)
    gradient_tresh = np.zeros(num_classes - 1)

    total_nll = 0.
    thresholds = _get_thresholds(deltas)

    # Vorhersage berechnen
    y_preds = X @ coefs

    # über jeden Datenpunkt iterieren
    for i in range(N):
        true_class = int(y[i])

        # Vorhersage extrahieren 
        y_pred = y_preds[i]

        # Ableitung der negativen Log-Likelihood nach Koeffizienten/Schwellenwerten für das Proportional-Odds-Modell (McCullagh 1980), siehe Appendix
        if true_class == 0:
            # Einfluss der Koeffizienten
            val_j = thresholds[true_class] - y_pred
            numerator_coef = _sigmoid_derivative(val_j)
            denominator_coef = _sigmoid(val_j)

            # Einfluss der Schwellenwerte
            numerator_thresh_j = _sigmoid_derivative(val_j)
            numerator_thresh_j_1 = 0.
            denominator_tresh_j = _sigmoid(val_j)
        elif true_class == num_classes - 1:
            # Einfluss der Koeffizienten
            val_j_1 = thresholds[true_class - 1] - y_pred
            numerator_coef = - _sigmoid_derivative(val_j_1)
            denominator_coef = 1. - _sigmoid(val_j_1)

            # Einfluss der Schwellenwerte
            numerator_thresh_j = 0
            numerator_thresh_j_1 = _sigmoid_derivative(val_j_1)
            denominator_tresh_j = 1. - _sigmoid(val_j_1)
        else:
            # Einfluss der Schwellenwerte
            val_j = thresholds[true_class] - y_pred
            val_j_1 = thresholds[true_class - 1] - y_pred
            numerator_coef = _sigmoid_derivative(val_j) - _sigmoid_derivative(val_j_1)
            denominator_coef = _sigmoid(val_j) - _sigmoid(val_j_1)

            numerator_thresh_j = _sigmoid_derivative(val_j)
            numerator_thresh_j_1 = _sigmoid_derivative(val_j_1)
            denominator_tresh_j = _sigmoid(val_j) - _sigmoid(val_j_1)

        # Verhinderung des Teilens durch 0
        if denominator_coef < 1e-5:
            denominator_coef = 1e-5
        if denominator_tresh_j < 1e-5:
            denominator_tresh_j = 1e-5

        # Gradienten berechnen mit Einfluss der Koeffizienten
        gradient_coef += (numerator_coef / denominator_coef) * X[i]

        # Gradienten berechnen mit Einfluss der Schwellenwerte
        if true_class < num_classes - 1:
            gradient_tresh[true_class] -= (numerator_thresh_j / denominator_tresh_j)
        if true_class > 0:
            gradient_tresh[true_class - 1] += (numerator_thresh_j_1 / denominator_tresh_j)

        # Verlustfunktion berechnen
        total_nll -= np.log(denominator_coef)

    # Threshold[i] hängt kumulativ von allen deltas[0..i] ab (siehe _get_thresholds); daher Kettenregel-Summation über alle nachfolgenden Schwellenwerte
    for m in range(num_classes - 1):
        gradient_delta[0] += gradient_tresh[m]
    for m in range(1, num_classes - 1):
        for n in range(m, num_classes - 1):
            gradient_delta[m] += gradient_tresh[n] * np.exp(deltas[m])

    return total_nll / N, np.concatenate((gradient_coef / N, gradient_delta / N)) # scipy.minimize mit jac=True erwartet Tupel (Skalar-Loss, Gradientenvektor)

class OrdinalLogRegModel():
    """
    Klasse zum Trainieren eines ordinalen Regressionsmodells.
    Verwendet die negative Log-Likelihood Funktion als Verlustfunktion.
    Berechnet den Gradienten basierend auf dem Proportional Odds Modell von Mc. Cullagh.
    """
    def __init__(self, num_classes, seed):
        """
        Args:
            num_classes: Anzahl der ordinalen Klassen (7 für Verweildauerkategorien)
            seed: Seed für Reproduzierbarkeit der Stichprobenziehung in compute_std_error
        """
        self.logger =       Logger(name = "Ordinal Log Reg Modell")
        self.num_classes =  num_classes
        self.seed =         seed

    def _initialize_params(self, X_train):
        """
        Initialisiert die Parameter.
        """
        _, p = X_train.shape
        coefs = np.zeros(p)
        deltas = np.zeros(self.num_classes - 1)

        return np.concatenate([coefs, deltas])

    def _wrapper(self, params, X, y):
        """
        Wrapper Funktion zum Berechnen des Gradienten und der Verlustfunktion.
        """
        _, p = X.shape
        coefs = params[:p]
        deltas = params[p:]

        nll, gradient = _calculate_gradient(coefs = coefs, deltas = deltas, num_classes = self.num_classes, X = X, y = y)

        return nll, gradient
    
    def compute_std_error(self, X, y, sample_size = 50_000):
        """
        Berechnet Standardfehler via Diagonalapproximation der Fisher-Information auf einer zufälligen Stichprobe.

        Args:
            X: Designmatrix
            y: Zielvariable
            sample_size: Stichprobengröße, Standard 50.000
        Returns:
            Array der Standardfehler je Parameter
        """

        # zieht zufällige Stichprobe
        rng = np.random.default_rng(seed=self.seed)
        idx = rng.choice(len(y), size=min(sample_size, len(y)), replace=False)
        X_sample = np.ascontiguousarray(X[idx], dtype=np.float64)
        y_sample = np.ascontiguousarray(y[idx], dtype=np.int64)

        # Transformiert die Daten mit einem Standardskalierer
        X_sample = self.scaler.transform(X_sample)

        N, p = X_sample.shape
        
        params = np.concatenate([self.coefs, self.deltas])
        
        n_params = len(params)
        eps = 1e-5

        _, grad = self._wrapper(params=params, X = X_sample, y = y_sample)

        # Standardfehler via Diagonalapproximation
        hess_diag = np.empty(n_params)
        for i in range(n_params):
            params_copy = params.copy()
            params_copy[i] += eps

            # Numerische Approximation der zweiten Ableitung (Diagonale der Hesse-Matrix) via finiter Differenz
            _, grad_addition = self._wrapper(params = params_copy, X = X_sample, y = y_sample)
            hess_diag[i] = (grad_addition[i] - grad[i]) / eps

        hess_diag_positive = np.where(np.abs(hess_diag) > 1e-10, hess_diag, 1e-10)
        variance_diag = 1. / (N * hess_diag_positive)

        return np.sqrt(np.clip(variance_diag, 0, None))
    
    def fit(self, X, y):
        """
        Trainiert das Proportional-Odds-Modell via L-BFGS-B.
        Standardisiert die Eingabedaten intern; Scaler wird für predict gespeichert.

        Args:
            X: Designmatrix (dense oder sparse)
            y: Ordinale Zielvariable als Integer
        """

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)

        # Daten skalieren für Standardisierung.
        self.scaler = StandardScaler(copy = False)
        X = self.scaler.fit_transform(X)

        N, p = X.shape
        self.params = self._initialize_params(X)

        # Verlustfunktion minimieren mit L-BFGS-B Solver
        res = minimize(
            fun=lambda params: self._wrapper(params, X, y),
            x0=self.params,
            method = "L-BFGS-B",
            tol = 1e-3,
            jac = True,
        )

        # finale Parameter extrahieren
        self.coefs = res.x[:p]
        self.deltas = res.x[p:]
        self.params = np.concatenate([self.coefs, self.deltas])
        self.thresholds = _get_thresholds(self.deltas)

        # Fehlermeldung bei unvollständiger Optimierung
        if not res.success:
            self.logger.warn("Optimierung nicht erfolgreich konvergiert.")

            if np.any(np.isnan(self.coefs)) or np.any(np.isnan(self.thresholds)):
                raise RuntimeError(f"Optimierung produzierte NaN-Parameter: {res.message}")
    
    def predict_proba(self, X_pred):
        """
        Sagt die Wahrscheinlichkeit für die Klassen der Verweildauer für gegebene Daten voraus.
        """
        if hasattr(X_pred, "toarray"):
            X_pred = X_pred.toarray()
            
        # Daten skalieren
        X_pred = self.scaler.transform(X_pred)
        N, p = X_pred.shape
        proba = np.zeros((N, self.num_classes))
        
        x_coef = X_pred @ self.coefs

        # Vorhersagen berechnen mit Randfällen
        for k in range(self.num_classes):
            if k == 0:
                val_j = self.thresholds[k] - x_coef
                proba[:, k] = _sigmoid(val_j)
            elif k == self.num_classes - 1:
                val_j_1 = self.thresholds[k - 1] - x_coef
                proba[:, k] = 1 - _sigmoid(val_j_1)
            else:
                val_j = self.thresholds[k] - x_coef
                val_j_1 = self.thresholds[k - 1] - x_coef
                proba[:, k] = _sigmoid(val_j) - _sigmoid(val_j_1)

        return proba
    
    def predict(self, X_pred):
        """
        Berechnet die vorhergesagte Verweildauer basierend auf den Wahrscheinlichkeiten der Klassen.
        """
        probas = self.predict_proba(X_pred)

        return np.argmax(probas, axis = 1)