import cv2
import numpy as np
import os

from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    f1_score, accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

try:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import ResNet50V2, DenseNet201, EfficientNetB7
    from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet
    from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

except Exception:
    from keras import layers, Model
    from keras.applications import ResNet50V2, DenseNet201, EfficientNetB7
    from keras.applications.resnet import preprocess_input as preprocess_resnet
    from keras.applications.densenet import preprocess_input as preprocess_densenet
    from keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

# ==================== CONFIG ====================
class Config:
    DATA_PATH = r"/kaggle/input/cervical-cancer-largest-dataset-sipakmed"
    
    USE_RESIZE = True
    TARGET_SIZE = (64, 64)
    
    DL_MODELS = ['resnet50v2', 'densenet201', 'efficientnetb7']
    DL_BATCH_SIZE = 32
    DL_POOLING_METHOD = 'avg'
    
    PCA_COMPONENTS = 0.95
    
    # ==================== CLASSIFIER TYPE ====================
    CLASSIFIER_TYPE = 'SVM'  
    # Options: 'SVM', 'RandomForest'
    
    # ==================== SVM PARAMETERS ====================
    SVM_TYPE = 'nu-SVC'  # 'C-SVC' or 'nu-SVC'
    SVM_C = 10
    SVM_NU = 0.7
    SVM_KERNEL = 'poly' # rbf, sigmoid, poly
    SVM_GAMMA = 'scale'
    SVM_DEGREE = 1
    SVM_COEF0 = 0.0

    AUTO_CLASS_WEIGHT = True  # Auto balance class weights
    RANDOM_STATE = 42
    CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# ==================== DATA LOADER ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data = []
        self.labels = []
        self.original_sizes = []  # Store original image sizes
        self.cropped_paths = []
    
    def load_data(self, data_path):
        i = 0

        for path in os.listdir(data_path):
            for p in os.listdir(os.path.join(data_path, path)):
                p = os.path.join(data_path, path, p, "CROPPED")
                self.cropped_paths.append(p)
        self.cropped_paths = sorted(self.cropped_paths)
        
        for p in self.cropped_paths:
            Class = os.listdir(p)
            for a in tqdm(Class):
                if(a[-1] == 'p'):
                    self.labels.append(i)
                    try:
                        image = cv2.imread(p + '/' + a)
                        image_from_array = Image.fromarray(image, 'RGB')
                        
                        # Store original size before resizing
                        original_size = image_from_array.size  # (width, height)
                        self.original_sizes.append(original_size)
                        
                        size_image = image_from_array.resize(self.config.TARGET_SIZE)
                        self.data.append(np.array(size_image))
                    except AttributeError:
                        print(" ")
            i += 1
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.original_sizes = np.array(self.original_sizes)

        X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
            self.data, self.labels, np.arange(len(self.data)),
            test_size=0.2,
            random_state=self.config.RANDOM_STATE,
            stratify=self.labels,
            shuffle=True
        )
        
        X_train_original_sizes = self.original_sizes[train_idx]
        X_test_original_sizes = self.original_sizes[test_idx]
        
        return X_train_raw, X_test_raw, y_train, y_test, X_train_original_sizes, X_test_original_sizes

# ==================== DEEP LEARNING FEATURE EXTRACTION ====================
class DeepLearningFeatureExtraction:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.preprocess_functions = {}
        self._load_models()
    
    def _add_pooling_and_top(self, base_model, model_name):
        if self.config.DL_POOLING_METHOD == 'avg':
            pooling = layers.GlobalAveragePooling2D()(base_model.output)
        elif self.config.DL_POOLING_METHOD == 'max':
            pooling = layers.GlobalMaxPooling2D()(base_model.output)
        else:
            raise ValueError(f"Unknown pooling method: {self.config.DL_POOLING_METHOD}")
        
        model_with_pooling = Model(inputs=base_model.input, outputs=pooling)
        return model_with_pooling
    
    def _load_models(self):
        input_shape = (self.config.TARGET_SIZE[0], self.config.TARGET_SIZE[1], 3)
        
        if 'resnet50v2' in self.config.DL_MODELS:
            base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
            model = self._add_pooling_and_top(base_model, 'resnet50v2')

            for layer in base_model.layers:
                layer.trainable = False
            
            self.models['resnet50v2'] = model
            self.preprocess_functions['resnet50v2'] = preprocess_resnet
        
        if 'densenet201' in self.config.DL_MODELS:
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
            model = self._add_pooling_and_top(base_model, 'densenet201')
            
            for layer in base_model.layers:
                layer.trainable = False
            
            self.models['densenet201'] = model
            self.preprocess_functions['densenet201'] = preprocess_densenet
        
        if 'efficientnetb7' in self.config.DL_MODELS:
            base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
            model = self._add_pooling_and_top(base_model, 'efficientnetb7')
            
            for layer in base_model.layers:
                layer.trainable = False
            
            self.models['efficientnetb7'] = model
            self.preprocess_functions['efficientnetb7'] = preprocess_efficientnet
        
        print(f"[DeepLearning] Models loaded: {list(self.models.keys())}")
        print(f"[DeepLearning] Pooling method: {self.config.DL_POOLING_METHOD}")
    
    def _prepare_batch(self, images, model_name):
        batch = []
        
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            img_processed = self.preprocess_functions[model_name](img.astype(np.float32))
            batch.append(img_processed)
        
        return np.array(batch)
    
    def extract_features(self, images):
        all_features = []
        
        for model_name in self.config.DL_MODELS:
            print(f"  Extracting {model_name} features...")
            batch = self._prepare_batch(images, model_name)
            
            features = self.models[model_name].predict(
                batch,
                batch_size=self.config.DL_BATCH_SIZE,
                verbose=0
            )
            
            print(f"    Shape: {features.shape}")
            all_features.append(features)
        
        concatenated_features = np.concatenate(all_features, axis=1)
        print(f"  Total concatenated features shape: {concatenated_features.shape}")
        
        return concatenated_features

# ==================== DIMENSIONALITY REDUCTION MODULE ====================
class DimensionalityReductionModule:
    def __init__(self, config):
        self.config = config
        self.pca = None
    
    def fit(self, features):
        print(f"\n[PCA] Fitting PCA...")
        print(f"  Input shape: {features.shape}")
        print(f"  Components: {self.config.PCA_COMPONENTS}")
        
        self.pca = PCA(
            n_components=self.config.PCA_COMPONENTS,
            random_state=self.config.RANDOM_STATE
        )
        reduced_features = self.pca.fit_transform(features)
        n_components = self.pca.n_components_
        
        if isinstance(self.config.PCA_COMPONENTS, float):
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"  Output shape: {reduced_features.shape}")
            print(f"  Actual components: {n_components}")
            print(f"  Explained variance: {explained_variance:.4f}")
        else:
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"  Output shape: {reduced_features.shape}")
            print(f"  Explained variance: {explained_variance:.4f}")
        
        return reduced_features
    
    def transform(self, features):
        if self.pca is None:
            raise ValueError("PCA not fitted yet! Call fit() first.")
        return self.pca.transform(features)

# ==================== CLASSIFIER MODULE ====================
class ClassifierModule:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.classifier = None
        self.is_fitted = False
        self.classes_ = None

    def _calculate_class_weights(self, y_train):
        if not self.config.AUTO_CLASS_WEIGHT:
            return None
        
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        class_weights = {}
        for class_label, count in zip(unique_classes, class_counts):
            # Inverse frequency weighting
            weight = total_samples / (len(unique_classes) * count)
            class_weights[class_label] = weight
        
        print(f"[ClassifierModule] Auto class weights calculated:")
        for class_label, weight in sorted(class_weights.items()):
            print(f"  Class {class_label}: {weight:.4f}")
        
        return class_weights
    
    def _create_classifier(self, class_weights=None):
        if self.config.CLASSIFIER_TYPE == 'SVM':
            if self.config.SVM_TYPE == 'C-SVC':
                class_weight = 'balanced' if self.config.AUTO_CLASS_WEIGHT else None
                self.classifier = SVC(
                    kernel=self.config.SVM_KERNEL,
                    C=self.config.SVM_C,
                    gamma=self.config.SVM_GAMMA,
                    degree=self.config.SVM_DEGREE,
                    coef0=self.config.SVM_COEF0,
                    probability=True,
                    class_weight=class_weight,
                    random_state=self.config.RANDOM_STATE,
                    verbose=0
                )
            else:
                class_weight = 'balanced' if self.config.AUTO_CLASS_WEIGHT else None
                self.classifier = NuSVC(
                    kernel=self.config.SVM_KERNEL,
                    nu=self.config.SVM_NU,
                    gamma=self.config.SVM_GAMMA,
                    degree=self.config.SVM_DEGREE,
                    coef0=self.config.SVM_COEF0,
                    probability=True,
                    class_weight=class_weight,
                    random_state=self.config.RANDOM_STATE,
                    verbose=0
                )
        elif self.config.CLASSIFIER_TYPE == 'RandomForest':
            self.classifier = RandomForestClassifier()
        
    def fit(self, X_train, y_train):
        class_weights = self._calculate_class_weights(y_train)
        self._create_classifier(class_weights)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.classifier.fit(X_train_scaled, y_train)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True

        train_pred = self.classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
        
        print(f"\n[Training Results]")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  F1-Score (weighted): {train_f1:.4f}")
        print(f"\n[Classifier] Training completed")
        print(f"  Classes: {len(self.classes_)}")
        print(f"  Samples trained: {X_train.shape[0]}")
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Classifier not fitted!")
        
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Classifier not fitted!")
        
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# ==================== VISUALIZATION MODULE ====================
class VisualizationModule:
    @staticmethod
    def plot_class_distribution(y_data, dataset_name, class_names=None):
        unique, counts = np.unique(y_data, return_counts=True)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique]
        else:
            class_names = [class_names[i] for i in unique]
        
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, counts)
        plt.title(f"Class Distribution - {dataset_name}")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"\n{dataset_name} Class Distribution:")
        for class_name, count in zip(class_names, counts):
            print(f"  {class_name}: {count}")

# ==================== TESTING CLASS ====================
class TestingModule:
    @staticmethod
    def test_first_sample_per_class(X_test_raw, y_test, X_test_original_sizes, 
                                     X_test_features, y_pred, classifier, scaler, pca_module,
                                     class_names, config):
        """Test first sample per class + Case 1 & 2 samples with same layout"""
        print("\n" + "="*70)
        print("TESTING: First Sample + Special Cases (Ambiguous & Misclassified)")
        print("="*70)
        
        proba = classifier.predict_proba(pca_module.transform(X_test_features))
        
        # ========== COLLECT SAMPLES FOR ALL 3 SECTIONS ==========
        
        # Section 1: First sample per class
        print("\n  Section 1: First sample from each class...")
        first_sample_indices = []
        # for class_idx in range(len(class_names)):
        #     first_idx = np.where(y_test == class_idx)[0]
        #     if len(first_idx) > 0:
        #         first_sample_indices.append(first_idx[0])
        
        # Section 2: Case 1 - Class 2/3 high near Class 1
        print("\n  Section 2: Class 2/3 probability high near Class 1...")
        ambiguous_indices = []
        
        for idx in range(len(y_test)):
            if y_test[idx] == y_pred[idx] and y_pred[idx] == 1:  # Correct prediction of class 1
                prob_class1 = proba[idx, 1]
                prob_class2 = proba[idx, 2] if len(class_names) > 2 else 0
                prob_class3 = proba[idx, 3] if len(class_names) > 3 else 0
                
                if (abs(prob_class1 - prob_class2) < 0.2 and prob_class2 > 0.1) or \
                   (abs(prob_class1 - prob_class3) < 0.2 and prob_class3 > 0.1):
                    ambiguous_indices.append(idx)
        
        case1_indices = ambiguous_indices[:5]
        if len(case1_indices) < 5:
            print(f"    WARNING: Only {len(case1_indices)} samples found (need 5)")
        
        # Section 3: Case 2 - Misclassified 1 per class, lowest conf
        print("\n  Section 3: Misclassified (1 per class, lowest confidence)...")
        case2_indices = []
        
        for class_idx in range(len(class_names)):
            misclassified_from_class = np.where((y_test == class_idx) & (y_test != y_pred))[0]
            
            if len(misclassified_from_class) > 0:
                confidences = [np.max(proba[idx]) for idx in misclassified_from_class]
                lowest_conf_idx = misclassified_from_class[np.argsort(confidences)[0]]
                case2_indices.append(lowest_conf_idx)
        
        # ========== COMBINE ALL SAMPLES ==========
        all_samples = [
            ("First Sample Per Class", first_sample_indices),
            ("Case 1: Class 2/3 High Near Class 1 (Predicted Correct)", case1_indices),
            ("Case 2: Misclassified (1 per class, Lowest Confidence)", case2_indices)
        ]
        
        # Calculate total rows needed
        total_samples = len(first_sample_indices) + len(case1_indices) + len(case2_indices)
        n_cols = 5
        
        fig, axes = plt.subplots(total_samples, n_cols, figsize=(22, 4*total_samples))
        if total_samples == 1:
            axes = axes.reshape(1, -1)
        
        row_idx = 0
        
        # ========== VISUALIZATION ==========
        for section_title, indices in all_samples:
            print(f"\n  Plotting {section_title}...")
            
            for sample_num, test_idx in enumerate(indices):
                image = X_test_raw[test_idx]
                true_label = y_test[test_idx]
                pred_label = y_pred[test_idx]
                original_size = X_test_original_sizes[test_idx]
                
                # Get predictions for this sample
                features_pca = pca_module.transform(X_test_features[test_idx:test_idx+1])
                sample_proba = proba[test_idx]
                
                # Display image
                axes[row_idx, 0].imshow(image)
                axes[row_idx, 0].set_title(f'Original Size: {original_size}')
                axes[row_idx, 0].axis('off')
                
                # Display info
                axes[row_idx, 1].axis('off')
                info_text = (
                    f"True Label: {class_names[true_label]}\n"
                    f"Predicted: {class_names[pred_label]}\n"
                    f"Correct: {'✓' if true_label == pred_label else '✗'}"
                )
                axes[row_idx, 1].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Top 3 probabilities - Bar chart
                top_3_idx = np.argsort(sample_proba)[-3:][::-1]
                axes[row_idx, 2].barh([class_names[i] for i in top_3_idx],
                                    [sample_proba[i] for i in top_3_idx],
                                    color=['green' if i == pred_label else 'blue' for i in top_3_idx])
                axes[row_idx, 2].set_xlabel('Probability')
                axes[row_idx, 2].set_title('Top 3 Probabilities (Bar Chart)')
                axes[row_idx, 2].set_xlim([0, 1])
                
                # Top 3 probabilities - Text
                axes[row_idx, 3].axis('off')
                prob_text = "Top 3 Predictions:\n\n"
                for i, idx in enumerate(top_3_idx):
                    prob_text += f"{i+1}. {class_names[idx]}: {sample_proba[idx]:.4f}\n"
                axes[row_idx, 3].text(0.1, 0.5, prob_text, fontsize=9, family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                                    verticalalignment='center')
                
                # Confidence info
                max_prob = np.max(sample_proba)
                axes[row_idx, 4].axis('off')
                conf_text = (
                    f"Max Probability: {max_prob:.4f}\n"
                    f"Confidence: {'High' if max_prob > 0.8 else 'Medium' if max_prob > 0.6 else 'Low'}"
                )
                axes[row_idx, 4].text(0.1, 0.5, conf_text, fontsize=10, family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                
                row_idx += 1
        
        plt.suptitle('Testing: First Sample + Special Cases', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig("visualize_sample.png", dpi=300, bbox_inches='tight')

        plt.show()
        
        # ========== DETAILED INFO ==========
        print("\n" + "="*80)
        print("DETAILED INFORMATION - SECTION 1: First Sample Per Class")
        print("="*80)
        
        for i, idx in enumerate(first_sample_indices):
            print(f"\nClass {i} ({class_names[i]}):")
            print(f"  True Label: {class_names[y_test[idx]]}")
            print(f"  Predicted: {class_names[y_pred[idx]]}")
            print(f"  Original Size: {X_test_original_sizes[idx]}")
            
            top_3_idx = np.argsort(proba[idx])[-3:][::-1]
            print(f"  Top 3 Probabilities:")
            for rank, class_idx in enumerate(top_3_idx, 1):
                print(f"    {rank}. {class_names[class_idx]}: {proba[idx, class_idx]:.4f}")
        
        print("\n" + "="*80)
        print("DETAILED INFORMATION - SECTION 2: Class 2/3 High Near Class 1")
        print("="*80)
        
        for i, idx in enumerate(case1_indices):
            print(f"\nSample {i+1}:")
            print(f"  True Label: {class_names[y_test[idx]]}")
            print(f"  Predicted: {class_names[y_pred[idx]]}")
            print(f"  Original Size: {X_test_original_sizes[idx]}")
            
            top_3_idx = np.argsort(proba[idx])[-3:][::-1]
            print(f"  Top 3 Probabilities:")
            for rank, class_idx in enumerate(top_3_idx, 1):
                print(f"    {rank}. {class_names[class_idx]}: {proba[idx, class_idx]:.4f}")
        
        print("\n" + "="*80)
        print("DETAILED INFORMATION - SECTION 3: Misclassified (Lowest Confidence)")
        print("="*80)
        
        for i, idx in enumerate(case2_indices):
            print(f"\nClass {i} ({class_names[i]}):")
            print(f"  True Label: {class_names[y_test[idx]]}")
            print(f"  Predicted: {class_names[y_pred[idx]]}")
            print(f"  Original Size: {X_test_original_sizes[idx]}")
            print(f"  Max Confidence: {np.max(proba[idx]):.4f}")
            
            top_3_idx = np.argsort(proba[idx])[-3:][::-1]
            print(f"  Top 3 Probabilities:")
            for rank, class_idx in enumerate(top_3_idx, 1):
                print(f"    {rank}. {class_names[class_idx]}: {proba[idx, class_idx]:.4f}")
    
    @staticmethod
    def plot_misclassified_samples(X_test_raw, y_test, X_test_original_sizes,
                                    y_pred, classifier, scaler, pca_module,
                                    X_test_features, class_names, config, top_n=10):
        """Plot top N misclassified samples with confusion pattern"""
        print(f"\n[Visualization] Plotting top {top_n} misclassified samples...")
        
        # Find misclassified samples
        misclassified_idx = np.where(y_test != y_pred)[0]
        n_misclassified = len(misclassified_idx)
        
        if n_misclassified == 0:
            print("  No misclassified samples!")
            return
        
        # Get confidence for each misclassification
        proba = classifier.predict_proba(pca_module.transform(X_test_features))
        
        misclassified_confidence = []
        for idx in misclassified_idx:
            pred_prob = proba[idx, y_pred[idx]]
            true_prob = proba[idx, y_test[idx]]
            confidence_diff = true_prob - pred_prob
            misclassified_confidence.append(confidence_diff)
        
        # Sort by confidence difference (most confident errors first)
        sorted_indices = np.argsort(misclassified_confidence)
        top_misclassified = misclassified_idx[sorted_indices[:top_n]]
        
        # Plot misclassified samples
        n_cols = 5
        n_rows = (len(top_misclassified) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten()
        
        for i, idx in enumerate(top_misclassified):
            image = X_test_raw[idx]
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            original_size = X_test_original_sizes[idx]
            
            axes[i].imshow(image)
            axes[i].set_title(
                f"TRUE: {class_names[true_label]}\n"
                f"PRED: {class_names[pred_label]}\n"
                f"Size: {original_size}",
                fontsize=10, color='red'
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(len(top_misclassified), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Confusion pattern analysis
        print(f"\n  Total misclassified: {n_misclassified} out of {len(y_test)} ({100*n_misclassified/len(y_test):.2f}%)")
        
        # Confusion matrix (only misclassified)
        cm_misclassified = confusion_matrix(y_test[misclassified_idx], y_pred[misclassified_idx],
                                           labels=range(len(class_names)))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_misclassified, annot=True, fmt='d', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Misclassification Count'})
        plt.title('Misclassification Pattern')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Per-class misclassification analysis
        print("\n  Misclassification per class:")
        for class_idx in range(len(class_names)):
            class_samples = np.sum(y_test == class_idx)
            class_misclassified = np.sum((y_test == class_idx) & (y_test != y_pred))
            if class_samples > 0:
                print(f"    {class_names[class_idx]}: {class_misclassified}/{class_samples} ({100*class_misclassified/class_samples:.2f}%)")

# ==================== EVALUATION MODULE ====================
class EvaluationModule:
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, y_true, y_pred, dataset_name="Test Set", class_names=None):
        print("\n" + "="*70)
        print(f"EVALUATION - {dataset_name}")
        print("="*70)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{'SUMMARY METRICS':<30}")
        print("-" * 50)
        print(f"{'Accuracy':<30} {accuracy:.4f}")
        print(f"{'F1-Score (macro)':<30} {f1_macro:.4f}")
        print(f"{'F1-Score (weighted)':<30} {f1_weighted:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\n{'PER-CLASS PERFORMANCE':<30}")
        print("-" * 80)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        header = f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}"
        print(header)
        print("-" * 80)
        
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                print(f"{class_name:<20} {precision[i]:<15.4f} {recall[i]:<15.4f} "
                      f"{f1_per_class[i]:<15.4f} {support[i]:<10.0f}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        ))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        metrics_dict = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class,
            'support': support,
            'confusion_matrix': cm,
            'class_names': class_names
        }
        
        return metrics_dict

# ==================== MAIN PIPELINE ====================
def main(config=None):
    if config is None:
        config = Config()
    
    # Initialize modules
    data_loader = DataLoader(config)
    dl_feature_extraction = DeepLearningFeatureExtraction(config)
    pca_module = DimensionalityReductionModule(config)
    classifier = ClassifierModule(config)
    evaluation = EvaluationModule(config)
    visualization = VisualizationModule()
    testing = TestingModule()
    scaler = MinMaxScaler()

    # ==================== PHASE 1: DATA LOADING ====================
    X_train_raw, X_test_raw, y_train, y_test, X_train_original_sizes, X_test_original_sizes = data_loader.load_data(config.DATA_PATH)
    visualization.plot_class_distribution(y_train, "Training Data", config.CLASSES)
    visualization.plot_class_distribution(y_test, "Test Data", config.CLASSES)
    
    # ==================== PHASE 2: FEATURE EXTRACTION (Deep Learning) ====================
    print("\nExtracting features from training images...")
    X_train_features = dl_feature_extraction.extract_features(list(X_train_raw))
    
    print("\nExtracting features from test images...")
    X_test_features = dl_feature_extraction.extract_features(list(X_test_raw))
    
    # ==================== PHASE 3: FEATURE NORMALIZATION ====================
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    # ==================== PHASE 4: REDUCE DIMENSIONALITY ====================
    X_train_pca = pca_module.fit(X_train_features)
    X_test_pca = pca_module.transform(X_test_features)
    
    # ==================== PHASE 5: TRAINING ====================
    classifier.fit(X_train_pca, y_train)
    
    # ==================== PHASE 6: EVALUATION ====================
    y_test_pred = classifier.predict(X_test_pca)
    
    test_metrics = evaluation.evaluate(
        y_test, y_test_pred,
        dataset_name="Test Set",
        class_names=config.CLASSES
    )
    
    # ==================== PHASE 7: TESTING - FIRST SAMPLE + SPECIAL CASES ====================
    testing.test_first_sample_per_class(
        X_test_raw, y_test, X_test_original_sizes,
        X_test_features, y_test_pred, classifier, scaler, pca_module,
        config.CLASSES, config
    )
    
    # # ==================== PHASE 8: MISCLASSIFICATION ANALYSIS ====================
    testing.plot_misclassified_samples(
        X_test_raw, y_test, X_test_original_sizes,
        y_test_pred, classifier, scaler, pca_module,
        X_test_features, config.CLASSES, config,
        top_n=config.VIZ_MISCLASSIFIED_TOP_N
    )
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("SUMMARY - CLASSIFICATION RESULTS")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1-Score (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main()