import numpy as np
import json
import os
from pathlib import Path
from itertools import product
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, log_loss
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, NuSVC
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
    SVM_NU = 0.1
    SVM_KERNEL = 'poly' # rbf, sigmoid, poly
    SVM_GAMMA = 'scale'
    SVM_DEGREE = 2
    SVM_COEF0 = 0.5

    AUTO_CLASS_WEIGHT = True  # Auto balance class weights
    RANDOM_STATE = 42
    CLASSES = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

# ==================== GPU UTILITY ====================
class GPUManager:
    """Manage GPU allocation for distributed grid search"""
    
    def __init__(self):
        self.gpus_available = self._detect_gpus()
        self.gpu_index = 0
    
    def _detect_gpus(self):
        """Detect available GPUs"""
        gpus = []
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpus = list(range(gpu_count))
                print(f"[GPUManager] Detected {gpu_count} GPU(s): {[f'GPU{i}' for i in gpus]}")
            else:
                print(f"[GPUManager] No GPUs detected, using CPU")
        except Exception as e:
            print(f"[GPUManager] GPU detection error: {e}")
        
        return gpus
    
    def get_gpu(self, combo_index):
        """Distribute GPU assignment across combinations"""
        if not self.gpus_available:
            return -1
        
        if len(self.gpus_available) == 1:
            return self.gpus_available[0]
        else:
            gpu_id = self.gpus_available[combo_index % len(self.gpus_available)]
            return gpu_id
    
    def set_gpu(self, gpu_id):
        """Set current GPU"""
        if gpu_id >= 0:
            try:
                torch.cuda.set_device(gpu_id)
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            except Exception as e:
                print(f"[GPUManager] Error setting GPU {gpu_id}: {e}")

# ==================== DATA LOADER ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data = []
        self.labels = []
        self.cropped_paths = []
    def load_data(self, data_path):
        i = 0

        for path in os.listdir(data_path):
            for p in os.listdir(os.path.join(data_path, path)):
                p = os.path.join(data_path, path, p, "CROPPED")
                self.cropped_paths.append(p)
        self.cropped_paths = sorted(self.cropped_paths)
        for p in self.cropped_paths:
            Class=os.listdir(p)
            for a in tqdm(Class):
                if(a[-1] == 'p'):
                    self.labels.append(i)
                    try:
                        image=cv2.imread(p + '/' + a)
                        image_from_array = Image.fromarray(image, 'RGB')
                        size_image = image_from_array.resize(self.config.TARGET_SIZE)
                        self.data.append(np.array(size_image))
                    except AttributeError:
                        print(" ")
            i+=1
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        return self.data, self.labels

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
            base_model = ResNet50V2(weights= 'imagenet', include_top=False, input_shape=input_shape)
            model = self._add_pooling_and_top(base_model, 'resnet50v2')

            for layer in base_model.layers:
                layer.trainable = False
            
            self.models['resnet50v2'] = model
            self.preprocess_functions['resnet50v2'] = preprocess_resnet
        
        if 'densenet201' in self.config.DL_MODELS:
            base_model = DenseNet201(weights= 'imagenet', include_top=False, input_shape=input_shape)
            model = self._add_pooling_and_top(base_model, 'densenet201')
            
            for layer in base_model.layers:
                layer.trainable = False
            
            self.models['densenet201'] = model
            self.preprocess_functions['densenet201'] = preprocess_densenet
        
        if 'efficientnetb7' in self.config.DL_MODELS:
            base_model = EfficientNetB7(weights= 'imagenet', include_top=False, input_shape=input_shape)
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

# ==================== AUTO-TUNER ====================
class AutoTuner:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.gpu_manager = GPUManager()
        self.best_results = None
        self.tuning_history = {}
    
    def tune_classifier(self, fold_data_list):
        params_grid = self._get_classifier_param_grid()
        combinations = self._generate_combinations(params_grid)
        
        combo_fold_scores = {}  # combo_id -> {combo: dict, fold_scores: []}
        
        print(f"\n[Auto-Tuner] Starting classifier tuning on {len(fold_data_list)} folds")
        print(f"[Auto-Tuner] Total combinations: {len(combinations)}")
        
        for combo_idx, combo in enumerate(combinations):
            gpu_id = self.gpu_manager.get_gpu(combo_idx)
            # if combo_idx+1 < 271: continue
            gpu_str = f"GPU{gpu_id}" if gpu_id >= 0 else "CPU"
            
            print(f"\n[Combo {combo_idx+1}/{len(combinations)}] {gpu_str}")
            print(f"  Params: {combo}")
            
            fold_scores = []
            fold_per_class_f1 = []
            
            for fold_idx, (X_train_pca, y_train, X_val_pca, y_val) in enumerate(fold_data_list):
                try:
                    clf = self._create_classifier(combo)
                    
                    # Scale and train
                    X_train_scaled = self.scaler.fit_transform(X_train_pca)
                    clf.fit(X_train_scaled, y_train)
                    
                    # Predict and evaluate
                    X_val_scaled = self.scaler.transform(X_val_pca)
                    val_pred = clf.predict(X_val_scaled)
                    
                    val_f1_macro = f1_score(y_val, val_pred, average='macro', zero_division=0)
                    per_class_f1 = f1_score(y_val, val_pred, average=None, zero_division=0)
                    
                    fold_scores.append(val_f1_macro)
                    fold_per_class_f1.append(per_class_f1.tolist())
                    
                    print(f"    Fold {fold_idx+1}: F1 Macro = {val_f1_macro:.4f}")
                    
                except Exception as e:
                    print(f"    Fold {fold_idx+1}: Error - {e}")
                    fold_scores.append(0.0)
                    fold_per_class_f1.append([0.0] * 5)
                    continue
            
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)
            mean_per_class_f1 = np.mean(fold_per_class_f1, axis=0).tolist()
            
            combo_fold_scores[combo_idx] = {
                'combo': combo,
                'fold_scores': fold_scores,
                'mean_f1': float(mean_f1),
                'std_f1': float(std_f1),
                'mean_per_class_f1': mean_per_class_f1,
                'combo_id': combo_idx
            }
            
            print(f"  Mean F1 (all folds): {mean_f1:.4f} ± {std_f1:.4f}")
            print(f"  Per-class F1: {[f'{f:.4f}' for f in mean_per_class_f1]}")
        
        ranked_results = sorted(
            combo_fold_scores.values(),
            key=lambda x: x['mean_f1'],
            reverse=True
        )
        
        for rank, result in enumerate(ranked_results, 1):
            result['rank'] = rank
        
        self.best_results = {
            'all_results': ranked_results,
            'best_combo': ranked_results[0]
        }

    def _generate_combinations(self, param_grid):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo_values in product(*values):
            combo_dict = {}
            for key, value in zip(keys, combo_values):
                combo_dict[key] = value
            combinations.append(combo_dict)
        
        return combinations
    
    def _get_classifier_param_grid(self):
        return {
            'SVM_TYPE': ['C-SVC', 'nu-SVC'],
            'SVM_C': [0.1, 0.5, 1, 10, 40] if 'C-SVC' in str(self.config.SVM_TYPE) else [1],
            'SVM_NU': [0.1, 0.3, 0.5, 0.7, 1] if 'nu-SVC' in str(self.config.SVM_TYPE) else [0.5],
            'SVM_DEGREE': [1, 2, 3],
            'SVM_GAMMA': ['scale', 'auto', 1e-1, 1, 3],
        }
    
    def _create_classifier(self, params):
        class_weight = 'balanced' if self.config.AUTO_CLASS_WEIGHT else None
        
        if params['SVM_TYPE'] == 'C-SVC':
            return SVC(
                kernel=self.config.SVM_KERNEL,
                C=params['SVM_C'],
                degree=params['SVM_DEGREE'],
                gamma=params['SVM_GAMMA'],
                probability=True,
                class_weight=class_weight,
                random_state=self.config.RANDOM_STATE,
                verbose=0
            )
        else:  # nu-SVC
            return NuSVC(
                kernel=self.config.SVM_KERNEL,
                nu=params['SVM_NU'],
                degree=params['SVM_DEGREE'],
                gamma=params['SVM_GAMMA'],
                probability=True,
                class_weight=class_weight,
                random_state=self.config.RANDOM_STATE,
                verbose=0
            )

    def save_ranking_to_jsonl(self, output_path='tuning_results_ranked.jsonl'):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for result in self.best_results['all_results']:
                f.write(json.dumps(result) + '\n')
        
        print(f"\nRanking saved to JSONL: {output_path}")
        return output_path
    
    def print_top_results(self, top_k=10):
        print(f"\n" + "="*70)
        print(f"TOP {top_k} BEST CONFIGURATIONS")
        print("="*70)
        
        for i, result in enumerate(self.best_results['all_results'][:top_k], 1):
            print(f"\n[Rank {i}] Mean F1: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
            print(f"  Fold Scores: {[f'{f:.4f}' for f in result['fold_scores']]}")
            print(f"  Per-class F1: {[f'{f:.4f}' for f in result['mean_per_class_f1']]}")
            print(f"  Params: {result['combo']}")

# ==================== MAIN WITH AUTOTUNER ====================
def main_with_tuning(config=None):
    if config is None:
        config = Config()

    data_loader = DataLoader(config)
    dl_extractor = DeepLearningFeatureExtraction(config)
    pca = DimensionalityReductionModule(config)
    scaler = MinMaxScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    tuner = AutoTuner(config)
    
    # ==================== PHASE 1: DATA LOADING ====================
    X, y = data_loader.load_data(config.DATA_PATH)
    
    fold_data_list = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Extract features
        print(f"\n[Extracting features for fold...]")
        X_train_features = dl_extractor.extract_features(X_train)
        X_val_features = dl_extractor.extract_features(X_val)
        
        # Scale data
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)
        
        # PCA reduction
        X_train_pca = pca.fit(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        fold_data_list.append((X_train_pca, y_train, X_val_pca, y_val))

    # ==================== PHASE 2: AUTO-TUNING (Cross-fold) ====================
    tuner.tune_classifier(fold_data_list)

    # ==================== PHASE 3: SAVE & PRINT RESULTS ====================
    tuner.save_ranking_to_jsonl()
    tuner.print_top_results(top_k=5)
    
    print(f"\n" + "="*70)
    print("BEST CONFIGURATION SUMMARY")
    print("="*70)
    best = tuner.best_results['best_combo']
    print(f"Mean F1 Macro: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")
    print(f"Fold Scores: {best['fold_scores']}")
    print(f"Per-class F1: {[f'{f:.4f}' for f in best['mean_per_class_f1']]}")
    print(f"Parameters: {best['combo']}")
    print("="*70)

if __name__ == "__main__":
    main_with_tuning()