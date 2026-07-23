"""
Rock-type classification for lunar mare basalts based on modal mineral abundances.

This module implements multiple classification approaches:
- Rule-based baseline classifier
- Gaussian Naive Bayes
- Support Vector Machine (RBF)
- Logistic Regression (multinomial)
- Random Forest
- Multilayer Perceptron
- XGBoost

Designed for reproducible evaluation and easy model comparison.
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Class index convention
# 0: Ilmenite basalt
# 1: Olivine basalt
# 2: Pigeonite basalt

CLASS_NAMES = [
    "Ilmenite Basalt",
    "Olivine Basalt",
    "Pigeonite Basalt"
]


def rule_based_classifier(row):
    """
    Heuristic rock-type classifier based on modal mineral abundances.

    Rules:
    - Olivine basalt: olivine > 5%
    - Ilmenite basalt: opaques > 7% and olivine ≤ 5%
    - Otherwise: pigeonite basalt
    """
    if row["Ol"] > 0.05:
        return 1
    elif row["Op"] > 0.07 and row["Ol"] <= 0.05:
        return 0
    else:
        return 2


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train and evaluate a classifier.

    Prints accuracy and classification reports for both training and test sets.

    Args:
        model: sklearn classifier instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display

    Returns:
        Trained model
    """
    model.fit(X_train, y_train)

    # Training evaluation
    y_pred_train = model.predict(X_train)
    print(f"\n###### {model_name} (Train) ######")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
    print(classification_report(
        y_train, y_pred_train, target_names=CLASS_NAMES
    ))

    # Test evaluation
    y_pred_test = model.predict(X_test)
    print(f"\n###### {model_name} (Test) ######")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
    print(classification_report(
        y_test, y_pred_test, target_names=CLASS_NAMES
    ))

    return model


def get_models(random_state=42):
    """
    Return a dictionary of classifiers configured for evaluation.

    Args:
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary mapping classifier names to sklearn/XGBoost classifier instances
    """
    return {
        "Gaussian Naive Bayes": GaussianNB(),

        "Support Vector Machine (RBF)": SVC(
            kernel="rbf",
            C=5,
            gamma="scale"
        ),

        "Logistic Regression": LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),

        "Multilayer Perceptron": MLPClassifier(
            hidden_layer_sizes=(15, 5),
            max_iter=1000,
            random_state=random_state
        ),

        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state
        )
    }


def evaluate_rule_based(df, minerals, labels):
    """
    Evaluate the rule-based classifier on a dataset.

    Args:
        df: DataFrame containing mineral abundance columns
        minerals: List of mineral column names to use for classification
        labels: Ground truth class labels
    """
    df = df.copy()
    df["RB_Pred"] = df[minerals].apply(rule_based_classifier, axis=1)

    print("\n###### Rule-Based Classifier ######")
    print(f"Accuracy: {accuracy_score(labels, df['RB_Pred']):.2f}")
    print(classification_report(
        labels, df["RB_Pred"], target_names=CLASS_NAMES
    ))


class RockClassifier:
    """
    Rock classifier that combines segmentation and classification.

    This class takes a trained segmentation model and multiple classification models
    to perform end-to-end rock type classification from SEM images.
    """

    def __init__(self, model, scaler, gnb, svm, lr, rf, mlp, xgb, pixel_size_dict, device, feature_order=None):
        """
        Initialize the rock classifier.

        Args:
            model: Trained segmentation model (UNet)
            scaler: StandardScaler fitted on training data
            gnb: Trained Gaussian Naive Bayes classifier
            svm: Trained SVM classifier
            lr: Trained Logistic Regression classifier
            rf: Trained Random Forest classifier
            mlp: Trained MLP classifier
            xgb: Trained XGBoost classifier
            pixel_size_dict: Dictionary mapping image names to pixel sizes
            device: torch device (cpu or cuda)
            feature_order: Optional list specifying feature column order
        """
        self.model = model
        self.scaler = scaler
        self.gnb = gnb
        self.svm = svm
        self.lr = lr
        self.rf = rf
        self.mlp = mlp
        self.xgb = xgb
        self.pixel_size_dict = pixel_size_dict
        self.device = device
        self.feature_order = feature_order

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def load_image(self, input_image_path):
        """
        Load and preprocess an input image.

        Args:
            input_image_path: Path to the input SEM image
        """
        self.img = cv2.imread(input_image_path)
        self.input_image = Image.open(input_image_path)
        self.transformed_input = self.transform(self.input_image).unsqueeze(0).to(self.device)
        base_name = os.path.basename(input_image_path).split('.')[0]
        self.pixel_size = self.pixel_size_dict[base_name]

    def predict(self):
        """Run segmentation prediction on the loaded image."""
        with torch.no_grad():
            self.prediction = self.model(self.transformed_input)
            _, self.predicted_class_map = self.prediction.max(dim=1)
            self.predicted_class_map_np = self.predicted_class_map.squeeze().cpu().numpy()

    def display_results(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title(f"Pixel size: {round(self.pixel_size)} µm/px")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.predicted_class_map_np, cmap='Spectral')

        plt.show()

    def classify(self):
        print("###### Rule-Based Classification #######")
        rb_result = self.rb_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(rb_result, "Rule-Based")
        print("\n###### GNB Classification ######")
        gnb_result = self.gnb_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(gnb_result, "GNB")
        print("\n###### SVM Classification ######")
        svm_result = self.svm_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(svm_result, "SVM")
        print("\n###### LR Classification ######")
        lr_result = self.lr_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(lr_result, "LR")
        print("\n###### RF Classification ######")
        rf_result = self.rf_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(rf_result, "RF")
        print("\n###### MLP Classification ######")
        mlp_result = self.mlp_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(mlp_result, "MLP")
        print("\n###### XGB Classification ######")
        xgb_result = self.xgb_modal_abundances(self.predicted_class_map_np, verbose=True)
        self.print_classification_result(xgb_result, "XGB")

    def print_classification_result(self, result, method):
        if result == 0:
            print(f"{method} classified as ilmenite basalt")
        elif result == 1:
            print(f"{method} classified as olivine basalt")
        elif result == 2:
            print(f"{method} classified as pigeonite basalt")

    def calculate_mineral_counts(self, predicted_class_map_np):
        """
        Count pixels for each mineral class in the segmentation map.

        Args:
            predicted_class_map_np: Segmentation map as numpy array

        Returns:
            Dictionary mapping mineral names to pixel counts
        """
        return {
            'void': np.count_nonzero(predicted_class_map_np == 0),
            'feni': np.count_nonzero(predicted_class_map_np == 1),
            'fes': np.count_nonzero(predicted_class_map_np == 2),
            'sp': np.count_nonzero(predicted_class_map_np == 3),
            'px': np.count_nonzero(predicted_class_map_np == 4),
            'pl': np.count_nonzero(predicted_class_map_np == 5),
            'gl': np.count_nonzero(predicted_class_map_np == 6),
            'ol': np.count_nonzero(predicted_class_map_np == 7),
            'ms': np.count_nonzero(predicted_class_map_np == 8),
            'na': np.count_nonzero(predicted_class_map_np == 9),
        }

    def calculate_mineral_percentages(self, mineral_counts, all_pixels):
        """
        Calculate percentage of each mineral.

        Args:
            mineral_counts: Dictionary of mineral counts
            all_pixels: Total number of pixels

        Returns:
            Dictionary mapping mineral names to percentages
        """
        return {k: v / all_pixels * 100 for k, v in mineral_counts.items()}

    def rb_modal_abundances(self, predicted_class_map_np, verbose=False):
        """
        Apply rule-based classification based on modal mineral abundances.

        Args:
            predicted_class_map_np: Segmentation map as numpy array
            verbose: If True, print mineral percentages

        Returns:
            int: Class label (0: Ilmenite, 1: Olivine, 2: Pigeonite)
        """
        all_pixels = predicted_class_map_np.size
        mineral_counts = self.calculate_mineral_counts(predicted_class_map_np)
        mineral_percentages = self.calculate_mineral_percentages(mineral_counts, all_pixels)

        if verbose:
            self.print_mineral_percentages(mineral_percentages)

        void_adjusted_total = 1 - mineral_percentages['void'] / 100
        if mineral_percentages['ol'] / void_adjusted_total > 5:
            return 1  # Olivine Basalt
        if mineral_percentages['sp'] / void_adjusted_total > 7 and mineral_percentages['ol'] / void_adjusted_total <= 5:
            return 0  # Ilmenite Basalt
        return 2  # Pigeonite Basalt

    def setup_modal_abundances(self, predicted_class_map_np, verbose=False):
        """
        Prepare modal abundance features from segmentation map for classification.

        Args:
            predicted_class_map_np: Segmentation map as numpy array
            verbose: If True, print additional information

        Returns:
            Scaled feature array ready for classifier input
        """
        mineral_counts = self.calculate_mineral_counts(predicted_class_map_np)

        norm_total = sum(mineral_counts.values()) - mineral_counts['void']
        if norm_total <= 0:
            return None

        new_sample = pd.DataFrame([{
            'Ol': mineral_counts['ol'] / norm_total,
            'Py': mineral_counts['px'] / norm_total,
            'Pl': mineral_counts['pl'] / norm_total,
            'Ms': mineral_counts['ms'] / norm_total,
            'Si': mineral_counts['gl'] / norm_total,
            'Op': (mineral_counts['feni'] + mineral_counts['fes'] + mineral_counts['sp']) / norm_total,
        }])

        # Align column order: scaler.feature_names_in_ > self.feature_order > current columns
        if hasattr(self.scaler, "feature_names_in_"):
            cols = list(self.scaler.feature_names_in_)
        elif getattr(self, "feature_order", None) is not None:
            cols = list(self.feature_order)
        else:
            cols = list(new_sample.columns)

        new_sample = new_sample.reindex(columns=cols, fill_value=0.0).astype(float)
        Xs = self.scaler.transform(new_sample)
        return Xs

    def gnb_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using Gaussian Naive Bayes."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.gnb.predict(Xs)[0])

    def svm_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using Support Vector Machine."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.svm.predict(Xs)[0])

    def lr_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using Logistic Regression."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.lr.predict(Xs)[0])

    def rf_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using Random Forest."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.rf.predict(Xs)[0])

    def mlp_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using Multilayer Perceptron."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.mlp.predict(Xs)[0])

    def xgb_modal_abundances(self, predicted_class_map_np, verbose=False):
        """Classify using XGBoost."""
        Xs = self.setup_modal_abundances(predicted_class_map_np, verbose=verbose)
        if Xs is None:
            return 2
        return int(self.xgb.predict(Xs)[0])

    def print_mineral_percentages(self, mineral_percentages):
        """Print mineral percentages."""
        for mineral, percentage in mineral_percentages.items():
            print(f"{mineral.capitalize()} Percentage: {percentage:.2f}%")

    def evaluate_single_image(self, input_image_path):
        """
        Evaluate a single image through segmentation and classification.

        Args:
            input_image_path: Path to the input SEM image

        Returns:
            Tuple of classification results from all methods:
            (rb_label, gnb_label, svm_label, lr_label, rf_label, mlp_label, xgb_label)
        """
        start_time = time.time()

        self.load_image(input_image_path)
        self.predict()

        base_name = os.path.basename(input_image_path).split('.')[0]

        rb_label = self.rb_modal_abundances(self.predicted_class_map_np)
        gnb_label = self.gnb_modal_abundances(self.predicted_class_map_np)
        svm_label = self.svm_modal_abundances(self.predicted_class_map_np)
        lr_label = self.lr_modal_abundances(self.predicted_class_map_np)
        rf_label = self.rf_modal_abundances(self.predicted_class_map_np)
        mlp_label = self.mlp_modal_abundances(self.predicted_class_map_np)
        xgb_label = self.xgb_modal_abundances(self.predicted_class_map_np)

        elapsed = time.time() - start_time

        label_map = {0: "Ilmenite", 1: "Olivine", 2: "Pigeonite"}

        print(f"[{base_name}] pixel: {self.pixel_size:.2f} µm → "
              f"RB: {label_map[rb_label]}, GNB: {label_map[gnb_label]}, SVM: {label_map[svm_label]}, "
              f"LR: {label_map[lr_label]}, RF: {label_map[rf_label]}, MLP: {label_map[mlp_label]}, "
              f"XGB: {label_map[xgb_label]}, Time: {elapsed:.2f}s", flush=True)

        return rb_label, gnb_label, svm_label, lr_label, rf_label, mlp_label, xgb_label


