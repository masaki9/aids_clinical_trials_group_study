import utils
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo


if __name__ == '__main__':
    # Fetch the dataset from UCI
    aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

    X = aids_clinical_trials_group_study_175.data.features
    y = aids_clinical_trials_group_study_175.data.targets.squeeze()

    random_state = 42
    scoring = 'f1_weighted'

    # Split the data into training and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Set up CV with stratification
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Set up a pipeline with a standard scaler and an SVM model with polynomial kernel and fit it to the training data
    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='poly')).fit(X_train, y_train)

    # Learning Curve with Pre-Tuned SVM
    utils.plot_learning_curve(svm_pipe, X_train, y_train, cv=cv, scoring=scoring, file_name='svm_learning_curve_pretuned.png', title='SVM Learning Curve (Polynomial Kernel, Pre-Tuned)', ylabel='F1 Score (Weighted)')

    # Generate a classification report for the SVM model
    utils.generate_classification_report(svm_pipe, X_test, y_test, title='SVM Classification Report', file_name='classification_report_pretuned.txt')

    # Validation Curve with C
    utils.plot_validation_curve(svm_pipe, X_train, y_train, param_name='svc__C', param_range=[0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0], cv=cv, scoring=scoring, file_name='svm_validation_curve_c.png', title='Validation Curve for SVM with Varying C', xlabel='C', ylabel='F1 Score (Weighted)')

    # Validation Curve with Degree
    utils.plot_validation_curve(svm_pipe, X_train, y_train, param_name='svc__degree', param_range=np.arange(1, 10), cv=cv, scoring=scoring, file_name='svm_validation_curve_degree.png', title='Validation Curve for SVM with Varying Degree', xlabel='Degree', ylabel='F1 Score (Weighted)')

    # Validation Curve with Coefficient
    utils.plot_validation_curve(svm_pipe, X_train, y_train, param_name='svc__coef0', param_range=[0.1, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0], cv=cv, scoring=scoring, file_name='svm_validation_curve_coef.png', title='Validation Curve for SVM with Varying Coefficient', xlabel='Coefficient', ylabel='F1 Score (Weighted)')

    # Grid Search

    # Define a parameter grid to search over
    param_grid = {
        'svc__C': np.arange(0.1, 2.1, 0.1),
        'svc__coef0': np.arange(0.5, 1.6, 0.1),
        'svc__degree': np.arange(1, 5, 1),
    }

    # Setup a GridSearchCV object
    grid_search = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the training data to the GridSearchCV object
    grid_search.fit(X_train, y_train)

    # Best parameters and best CV score found during the grid search
    print("\nBest parameters:", grid_search.best_params_)
    print("Best CV score: {:.2f}".format(grid_search.best_score_))

    # Get the best estimator, which has already been trained on the training set
    best_estimator = grid_search.best_estimator_

    # Evaluate the best estimator on the test set and generate a classification report
    utils.generate_classification_report(best_estimator, X_test, y_test, title='SVM Classification Report', file_name='classification_report.txt')

    # Learning curve SVM with the tuned parameters
    utils.plot_learning_curve(best_estimator, X_train, y_train, cv, scoring, file_name='svm_learning_curve_posttuned.png', title='SVM Learning Curve (Polynomial Kernel, Post-Tuned)', ylabel='F1 Score (Weighted)')
