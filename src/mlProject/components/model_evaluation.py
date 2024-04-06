import os
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score
from mlProject.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        cm = confusion_matrix(actual, pred)
        accuracy = accuracy_score(actual, pred)
       
        return cm , accuracy
    def k_fold_cross_validation(self, estimator, X, y, cv=10):
        kfc = cross_val_score(estimator=estimator, X = X, y = y, cv=cv)

        return kfc
        
    def save_results(self):
        train_data =pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (cm , accuracy) = self.eval_metrics(test_y, predicted_qualities)
        
        kfc = self.k_fold_cross_validation(model,train_x,train_y)
        
        mean_accuracy = kfc.mean() * 100
        std_deviation = kfc.std() * 100
        
        cm = cm.tolist()
        kfc = kfc.tolist()
        # Saving metrics as local
        scores = {"confusion_matrix": cm,
                  "accuracy_score": accuracy,
                  "mean accuracy": mean_accuracy,
                  "standard deviation": std_deviation }
        save_json(path=Path(self.config.metric_file_name), data=scores)