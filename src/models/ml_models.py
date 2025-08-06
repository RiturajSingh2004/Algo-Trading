from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from ..utils.logger import setup_logger

def train_ensemble_models(df):
    """Train multiple ML models and return the best performing one."""
    logger, _ = setup_logger()
    
    if df is None or df.empty:
        logger.error("No data provided for ML training")
        return None, 0, {}
    
    logger.info("Training ensemble ML models")
    
    try:
        # Create target variable
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Define features
        ml_features = [
            'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'WILLR_14',
            'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'ATRr_14', 'AD', 'OBV',
            'Price_Change', 'Volume_Change', 'High_Low_Pct', 'Open_Close_Pct',
            'Distance_to_Support', 'Distance_to_Resistance'
        ]
        
        # Filter available features
        available_features = [f for f in ml_features if f in df.columns]
        
        if len(available_features) < 5:
            logger.warning("Insufficient features for ML training")
            return None, 0, {}
        
        # Prepare data
        ml_df = df[available_features + ['Target']].dropna()
        X = ml_df[available_features]
        y = ml_df['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            )
        }
        
        results = {}
        best_model = None
        best_accuracy = 0
        
        for name, model in models.items():
            try:
                if name == 'Logistic Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'model': model,
                    'scaler': scaler if name == 'Logistic Regression' else None
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {report['weighted avg']['f1-score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            logger.error("No models were successfully trained")
            return None, 0, {}
        
        # Get feature importance
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            importance_dict = dict(zip(available_features, best_model.feature_importances_))
            feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
        
        logger.info(f"Best model accuracy: {best_accuracy:.3f}")
        
        return best_model, round(best_accuracy * 100, 2), {
            'model_results': results,
            'feature_importance': feature_importance,
            'features_used': available_features
        }
        
    except Exception as e:
        logger.error(f"Error in ML model training: {e}")
        return None, 0, {}
