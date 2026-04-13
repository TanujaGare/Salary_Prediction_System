import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import scipy.stats as stats

# Initialize Flask App
app = Flask(__name__, template_folder='.')
CORS(app)

DATA_PATH = 'Salary_Data.csv'

def calculate_adjusted_r2(r2, n, p):
    """Calculates Adjusted R-squared."""
    if n - p - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_model(model, X_train, y_train, model_name=""):
    """Evaluates a model and returns standard metrics."""
    # Assuming we train and evaluate on the same dataset given the small size (30 rows).
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    n = len(y_train)
    p = X_train.shape[1] if len(X_train.shape) > 1 else 1
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    
    # 5-Fold Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    
    return {
        'name': model_name,
        'r2': float(r2),
        'adj_r2': float(adj_r2),
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred.tolist()
    }

@app.route('/')
def index():
    """Renders the single page application frontend."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Runs the full dataset analysis pipeline: 
    Stats, Preprocessing, EDA, Models, Residuals, Bootstrapping, PCA.
    """
    try:
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save('uploaded_data.csv')
            data_path = 'uploaded_data.csv'
        else:
            data_path = DATA_PATH

        if not os.path.exists(data_path):
            return jsonify({'error': f'Dataset not found at {data_path}'}), 404
        
        df = pd.read_csv(data_path)
        df = df.dropna()
        if len(df.columns) < 2:
            return jsonify({'error': 'Dataset must have at least 2 columns.'}), 400

        col_x = df.columns[0]
        col_y = df.columns[1]

        X = df[[col_x]].values
        y = df[col_y].values

        # [1] DATA LOADING & DESCRIPTIVE STATISTICS
        desc_stats = {}
        for col in [col_x, col_y]:
            desc_stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'mode': float(df[col].mode()[0]) if not df[col].mode().empty else float(df[col].mean()),
                'std': float(df[col].std()),
                'variance': float(df[col].var()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
            
        # [2] DATA PREPROCESSING
        missing_values = df.isnull().sum().to_dict()
        
        outliers_info = {}
        for col in [col_x, col_y]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': int(len(outliers))
            }
            
        std_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler()
        X_zscore = std_scaler.fit_transform(X)
        X_minmax = min_max_scaler.fit_transform(X)
        
        norm_example = {
            'original': float(X[0][0]),
            'zscore': float(X_zscore[0][0]),
            'minmax': float(X_minmax[0][0])
        }

        # [3] EDA
        pearson_corr = float(df[col_x].corr(df[col_y]))
        cov_matrix = df[[col_x, col_y]].cov().values.tolist()
        scatter_data = [{'x': float(row[col_x]), 'y': float(row[col_y])} for _, row in df.iterrows()]

        # [4] MODELS
        models_results = []
        
        # Simple Linear
        lr = LinearRegression()
        lr.fit(X, y)
        models_results.append(evaluate_model(lr, X, y, "Linear Regression"))
        
        # Poly Degree 2
        poly2 = PolynomialFeatures(degree=2)
        X_poly2 = poly2.fit_transform(X)
        lr_poly2 = LinearRegression()
        lr_poly2.fit(X_poly2, y)
        models_results.append(evaluate_model(lr_poly2, X_poly2, y, "Polynomial Regression (Deg 2)"))
        
        # Poly Degree 3
        poly3 = PolynomialFeatures(degree=3)
        X_poly3 = poly3.fit_transform(X)
        lr_poly3 = LinearRegression()
        lr_poly3.fit(X_poly3, y)
        models_results.append(evaluate_model(lr_poly3, X_poly3, y, "Polynomial Regression (Deg 3)"))
        
        # Ridge (L2)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        models_results.append(evaluate_model(ridge, X, y, "Ridge Regression (L2)"))
        
        # Lasso (L1)
        lasso = Lasso(alpha=1.0)
        lasso.fit(X, y)
        models_results.append(evaluate_model(lasso, X, y, "LASSO Regression (L1)"))
        
        best_model_data = max(models_results, key=lambda x: x['r2'])

        # [5] RESIDUAL ANALYSIS
        best_preds = np.array(best_model_data['predictions'])
        residuals = y - best_preds
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        
        residual_plot_data = [{'x': float(p), 'y': float(r)} for p, r in zip(best_preds, residuals)]
        
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        qq_plot_data = [{'x': float(th), 'y': float(r)} for th, r in zip(theoretical_quantiles, sorted_res)]

        # [6] BOOTSTRAPPING
        n_bootstraps = 1000
        bootstrapped_slopes = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_b = X[indices]
            y_b = y[indices]
            lr_b = LinearRegression()
            lr_b.fit(X_b, y_b)
            bootstrapped_slopes.append(lr_b.coef_[0])
            
        lower_bound = float(np.percentile(bootstrapped_slopes, 2.5))
        upper_bound = float(np.percentile(bootstrapped_slopes, 97.5))
        mean_slope = float(np.mean(bootstrapped_slopes))

        # [7] PCA
        X_pca = np.column_stack((X_zscore.flatten(), std_scaler.fit_transform(y.reshape(-1, 1)).flatten()))
        pca = PCA(n_components=2)
        pca.fit(X_pca)
        explained_variance = pca.explained_variance_ratio_.tolist()

        return jsonify({
            'dataset_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'feature_names': list(df.columns),
                'col_x': col_x,
                'col_y': col_y
            },
            'descriptive_stats': desc_stats,
            'missing_values': missing_values,
            'outliers_info': outliers_info,
            'normalization_example': norm_example,
            'eda': {
                'pearson_corr': pearson_corr,
                'cov_matrix': cov_matrix,
                'scatter_data': scatter_data
            },
            'models_results': models_results,
            'best_model': best_model_data['name'],
            'residual_analysis': {
                'mean': res_mean,
                'std': res_std,
                'residual_plot_data': residual_plot_data,
                'qq_plot_data': qq_plot_data
            },
            'bootstrapping': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'mean_slope': mean_slope
            },
            'pca': {
                'explained_variance': explained_variance
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predicts target variable given the feature value and provides a CI."""
    try:
        data = request.json
        exp = float(data.get('experience', 0))
        
        data_path = 'uploaded_data.csv' if os.path.exists('uploaded_data.csv') else DATA_PATH

        if not os.path.exists(data_path):
            return jsonify({'error': 'Dataset missing'}), 404
            
        df = pd.read_csv(data_path)
        if len(df.columns) < 2:
            return jsonify({'error': 'Dataset must have at least 2 columns'}), 400
            
        col_x = df.columns[0]
        col_y = df.columns[1]

        X = df[[col_x]].values
        y = df[col_y].values
        
        # Normally you would load a saved model, but training it on the fly is fine for 30 rows.
        lr = LinearRegression()
        lr.fit(X, y)
        pred = lr.predict([[exp]])[0]
        
        y_pred_all = lr.predict(X)
        std_err = np.std(y - y_pred_all)
        ci_lower = pred - 1.96 * std_err
        ci_upper = pred + 1.96 * std_err
        
        return jsonify({
            'prediction': float(pred),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'model_used': 'Linear Regression'
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
