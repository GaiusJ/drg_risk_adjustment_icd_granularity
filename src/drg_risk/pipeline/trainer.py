import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from scipy.sparse import csr_matrix
import gc
import pandas as pd

from drg_risk.utils.logger import Logger
from drg_risk.evaluation.evaluator import Evaluator
from drg_risk.models.ordinal_log_regression_model import OrdinalLogRegModel
from drg_risk.evaluation.score_postprocessing import Score_PostProcessor


class Trainer():
    """
    Verwaltet Datenladen, Modellinitialisierung, Kreuzvalidierung und Evaluation für Mortalitäts- (SGD/logistisch) und Verweildauermodelle (Proportional-Odds).
    """
    def __init__(self, loader, output_var, icd_structure, use_sdx = True, load_columns = None, load_subset = None, load_shuffle = True, seed = 42, class_weight = None, input_var = "codes", kovarianten = None):
        """
        Args:
            loader: DataLoader-Instanz mit geladenen Campus-File-Daten
            output_var: Zielvariable ('entl_grd' für Mortalität, 'typ_vwd' für Verweildauer)
            icd_structure: 'group' oder 'chapter'
            use_sdx: Ob Nebendiagnosen verwendet werden
            load_columns: Basisspalten die geladen werden (ICD-Spalten werden ergänzt)
            load_subset: Optionaler Filter, z.B. {'aufn_anl': ['N']}
            load_shuffle: Ob Daten vor dem Training gemischt werden
            seed: Seed für Reproduzierbarkeit
            class_weight: Klassengewichtung für SGD (None = ungewichtet)
            input_var: Art der Eingabevariablen ('codes', 'charlson', 'elixhauser')
            kovarianten: Liste demografischer Kovariaten oder None
        """
        self.logger =                       Logger(name = "trainer")
        self.loader =                       loader
        self.output_var =                   output_var
        self.loader.load_data(output_var = self.output_var)
        
        self.icd_structure =                icd_structure
        self.seed =                         seed
        self.use_sdx =                      use_sdx
        self.subset =                       load_subset
        self.data =                         self.loader.get_data(use_sdx=use_sdx, columns = load_columns, subset = self.subset, shuffle = load_shuffle, seed = seed)
        self.class_weight =                 class_weight
        self.input_var =                    input_var
        self.kovarianten =                  kovarianten

        # Wenn Kovarianten und Subset gleichzeitig verwendet werden, entfernen wir Kovarianten, die im Subset fixiert sind, um Multikollinearität zu vermeiden
        if self.kovarianten and self.subset:
            conflicting = [k for k in self.kovarianten if k in self.subset and len(self.subset[k]) == 1]
            if conflicting:
                self.logger.warn(f"Kovariante(n) {conflicting} entfernt, da in subset.")
                self.kovarianten = [k for k in self.kovarianten if k not in conflicting]

        # Wenn Subgruppen verwendet wurden
        if self.subset:
            key = list(self.subset.keys())[0]
            value = list(self.subset.values())[0]
            value_str = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            self.subset_str = f"subset {key}={value_str}"
        else:
            self.subset_str = "subset=gesamt"

        # Daten mithilfe des Loaders laden
        self._build_data()

        self.logger.info("Trainer erfolgreich initialisiert")
        
    def _build_data(self):
        """
        Daten mithilfe des Dataloaders laden und zur Verwendung während des Trainierens aufbereiten.
        """
        self.logger.info(f"Erstelle Dataset, Verwendung von {self.icd_structure}")

        # Verwendete Variablen definieren
        if self.input_var == "codes":
            if self.icd_structure == "group":
                self.feature_cols = [col for col in self.data.columns if col.startswith("icd_group")]
            elif self.icd_structure == "chapter":
                self.feature_cols = [col for col in self.data.columns if col.startswith("icd_chapter")]
            else:
                raise ValueError("icd_structure muss icd_group oder icd_chapter sein.")
        elif self.input_var == "charlson":
            self.feature_cols = [f"charlson_{self.icd_structure}"]
        elif self.input_var == "elixhauser":
            self.feature_cols = [f"elixhauser_{self.icd_structure}"]
        else:
            raise ValueError("input_var muss codes, charlson oder elixhauser sein.")

        # Wenn Kovarianten verwendet werden
        if self.kovarianten:
            self.feature_cols = self.feature_cols + self.kovarianten
            if "sex" in self.kovarianten:
                # Umwandeln von Buchstaben in Zahlen
                self.data["sex"] = self.data["sex"].map({"m": 0, "w": 1})
            if "aufn_anl" in self.kovarianten:
                # In Multi-Hot-Kodierung umwandeln
                self.data = pd.get_dummies(self.data, columns=["aufn_anl"], drop_first=True)
                dummy_cols = [col for col in self.data.columns if col.startswith("aufn_anl")]
                self.feature_cols = [c for c in self.feature_cols if c != "aufn_anl"] + dummy_cols

        # Anzahl der Parameter berechnen
        if self.output_var == "entl_grd":
            self.num_params = len(self.feature_cols) + 1        # parameter = intercept + für jede Gruppe / Kapitel
        elif self.output_var == "typ_vwd":
            self.num_params = len(self.feature_cols) + (7 - 1)  # parameter = für jede Gruppe / Kapitel + für jeden Threshold (num_classes - 1) (anstelle des Intercepts)

        # Datentyp gewährleisten, Zuweisung der Variablen
        X = self.data[self.feature_cols].values.astype(np.float32)
        if self.output_var == "entl_grd":
            y = (self.data[self.output_var] == 5).astype(np.float32)
            self.X = X
            self.y = y.values
        elif self.output_var == "typ_vwd":
            y = self.data[self.output_var].astype(np.int8) - 1
            self.X = np.array(X)
            self.y = np.array(y)

        self.logger.info("Daten erfolgreich geladen.")
    
    def _build_model(self):
        """
        Initialisiert das Modell anhand der Hyperparameter und Zielvariable.
        """

        # Mortalitätsvorhersage
        if self.output_var == "entl_grd":
            self.logger.info(f"Initialisiere Modell mit loss = log_loss, penalty = None, class_weight = {self.class_weight}, output_var = {self.output_var} und random_state = {self.seed}")
            sgd = SGDClassifier(
                loss =                  "log_loss", # Verwendung von log_loss für binäre Klassifikation
                penalty =               None, # Keine Regularisierung, da EVP >= 2500, Steyerberg: min. 50
                learning_rate =         "adaptive", # Adaptive Lernrate, um die Konvergenz zu beschleunigen
                eta0 =                  .05, # Startwert für die Lernrate, da adaptive Lernrate verwendet wird, um die Konvergenz zu beschleunigen
                random_state =          self.seed, # Seed für Reproduzierbarkeit
                fit_intercept =         True, # Füge Intercept hinzu, da die Daten nicht zentriert sind
                max_iter =              10_000, # Maximale Anzahl an Iterationen, um Konvergenz zu erreichen
                tol =                   1e-6, # Abbruch, wenn sich die Kostenfunktion nicht mehr verbessert
                n_iter_no_change =      5, # Abbruch, wenn sich die Kostenfunktion nicht mehr verbessert
                shuffle =               True, # Daten werden vor jedem Epoch gemischt, um Overfitting zu vermeiden
                verbose =               1, # Ausgabe von Informationen während des Trainings
                early_stopping =        True, # Abbruch, wenn sich die Validierungsgenauigkeit nicht verbessert
                validation_fraction =   1/9, # 1/9 der Trainingsdaten = 1/10 der Gesamtdaten werden für die Validierung verwendet, da training data = 9/10
                class_weight =          self.class_weight, # None
                warm_start =            False, # keine Wiederverwendung der Koeffizienten, da das Modell auf dem gesamten Datensatz trainiert wurde
            )
            self.model = Pipeline([
                ('scaler', StandardScaler(copy = False, with_mean=False)),
                ('sgd', sgd)
            ])
        # Verweildauervorhersage
        elif self.output_var == "typ_vwd":
            self.logger.info(f"Initialisiere Modell mit loss = log_loss, output_var = {self.output_var} und random_state = {self.seed}")
            self.model = OrdinalLogRegModel(num_classes=7, seed = self.seed)
        self.logger.info("Modell erfolgreich initialisiert.")

    def _train(self):
        """
        Funktion zum Initialisieren und Trainieren des Models.
        """
        self._build_model()
        self.logger.info(f"Trainere Modell.")

        # Sparse-Format für Speichereffizienz beim SGD-Training (viele Multi-Hot-kodierte ICD-Spalten sind größtenteils 0)
        X_train_sparse  = csr_matrix(self.X_train)

        self.model.fit(X_train_sparse, self.y_train)

        gc.collect()
        
        self.logger.info(f"Modell trainiert.")

    # Cross Validation
    def cross_validate(self, n_splits=10):
        """
        Funktion zum Kreuzvalidieren eines Modells.
        """
        
        self.logger.info(f"Kreuzvalidiere Modell für {self.output_var} mit {n_splits} Splits.")

        # Verwendung von stratifizierter Kreuzvalidierung
        strat_k_fold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = self.seed)

        self.y_test_fold_combined = []
        self.y_test_fold_combined_pred_probas = []

        # Evaluator intialisieren
        self.evaluator = Evaluator(
            output_var=self.output_var,
            use_sdx=self.use_sdx,
            feature_cols = self.feature_cols,
            icd_structure=self.icd_structure,
            subset=self.subset,
            input_var = self.input_var,
            kovarianten = self.kovarianten
        )

        # Laufzeitmessung
        start_time = pd.Timestamp.now()

        # Über folds iterieren
        for fold, (train_index, test_index) in enumerate(strat_k_fold.split(self.X, self.y)):
            # Daten zuweisen
            self.X_train =  self.X[train_index]
            self.X_test =   self.X[test_index]
            self.y_train =  self.y[train_index]
            self.y_test =   self.y[test_index]

            self._train()

            # Testwerte vorhersagen
            self.y_train_pred_proba = self.model.predict_proba(self.X_train)
            self.y_test_pred_proba  = self.model.predict_proba(self.X_test)
            self.y_test_pred = self.model.predict(self.X_test)

            self.y_test_fold_combined.append(self.y_test)

            if self.output_var == "entl_grd":
                self.y_train_pred_proba = self.y_train_pred_proba[:, 1]
                self.y_test_pred_proba = self.y_test_pred_proba[:, 1]

            self.y_test_fold_combined_pred_probas.append(self.y_test_pred_proba)

            # Bei Mortalität wird SGD in einer Pipeline verwendet (wegen Scaler) - Evaluator braucht direkten Modellzugriff für Koeffizienten
            actual_model = self.model['sgd'] if isinstance(self.model, Pipeline) else self.model

            # Modell evaluieren
            scores = self.evaluator.evaluate_model_out_sample(
                fold, self.y_train, self.y_train_pred_proba, 
                self.y_test, self.y_test_pred_proba, self.y_test_pred, 
                self.X_test, self.feature_cols, actual_model
            )

            # Metriken speichern
            self.evaluator.save_out_sample_scores(fold = fold, scores = scores)

            self.logger.info(f"Fold {fold + 1}/{n_splits} abgeschlossen. Scores gespeichert.")

        # Model auf gesamten Datensatz evaluieren
        scores = self.evaluator.evaluate_model_out_sample_aggregated(self.y_test_fold_combined, self.y_test_fold_combined_pred_probas)
        # Metriken speichern
        self.evaluator.save_aggregated_scores_out_sample(scores)

        # Modell auf gesamten Datensatz trainieren und evaluieren
        self._train_in_sample()

        self.score_postprocessor = Score_PostProcessor(
            output_var =        self.output_var,
            use_sdx =           self.use_sdx,
            icd_structure =     self.icd_structure,
            subset =            self.subset,
            n_folds =           n_splits,
            input_var =         self.input_var,
            kovarianten =       self.kovarianten
        )

        # Bericht erstellen
        self.score_postprocessor.generate_score_report(start_time, f"Trainingsdaten: {self.X_train.shape[0]} Samples, {self.X_train.shape[1]} Features.")



    def _train_in_sample(self):
        """
        Trainiert und Evaluiert das Modell auf dem gesamten Datensatz
        """
        self.logger.info("Trainere das Modell auf dem gesamten Datensatz.")
        self.X_train = self.X # Gesamten Datensatz verwenden
        self.y_train = self.y

        # Trainieren
        self._train()

        # Vorhersage
        self.y_train_pred_proba = self.model.predict_proba(self.X_train)

        if self.output_var == "entl_grd":
                self.y_train_pred_proba = self.y_train_pred_proba[:, 1]

        # Evaluieren
        actual_model = self.model['sgd'] if isinstance(self.model, Pipeline) else self.model
        scores = self.evaluator.evaluate_model_in_sample(self.y_train, self.y_train_pred_proba, self.num_params, actual_model, self.X_train)

        self.evaluator.save_in_sample_scores(scores = scores)

