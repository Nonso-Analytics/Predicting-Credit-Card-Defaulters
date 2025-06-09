import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import (
    f1_score, classification_report,
    average_precision_score, precision_recall_curve,
    roc_curve, auc, precision_score, recall_score, roc_auc_score,
    confusion_matrix, accuracy_score
)

import warnings
warnings.filterwarnings('ignore')

def main():
    st.title("Credit Default Predictor")
    st.sidebar.title("Credit Default Predictor")
    st.markdown("Identify potential credit defaulters for a Taiwanese bank")
    st.sidebar.markdown("This application uses machine learning to predict credit defaulters based on customer financial data.")

    # Define model paths (adjust these paths to match your model locations)
    MODEL_PATHS = {
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl", 
        "Logistic Regression": "logistic_regression_model.pkl"
    }
    
    # Check which models are available
    available_models = {}
    for model_name, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            available_models[model_name] = model_path
        else:
            st.sidebar.warning(f"{model_name} model not found at: {model_path}")

    if not available_models:
        st.error("No pre-trained models found! Please ensure model files are in the correct directory.")
        st.markdown("""
        ### Expected Model Locations:
        - `random_forest_model.pkl`
        - `xgboost_model.pkl` 
        - `logistic_regression_model.pkl`
        
        Make sure these files are in the same directory as your app.py file.
        """)
        return
    
    # Display available models
    st.sidebar.success(f"{len(available_models)} model(s) loaded successfully!")
    for model_name in available_models.keys():
        st.sidebar.info(f"✓ {model_name}")
    
    # Data file upload section
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    @st.cache_data
    def load_model(model_path):
        """Load a pre-trained model from pickle file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            st.error(f"Error loading model from {model_path}: {str(e)}")
            return None

    @st.cache_data
    def feature_engineering(df):
        """Apply the same feature engineering as used during training"""
        try:
            # Make a copy to avoid modifying original
            cc_df = df.copy()
            cc_df.columns = cc_df.columns.str.lower()
            
            # Feature Engineering (same as training)
            cc_df['avg_bill_amt'] = cc_df[[f'bill_amt{i}' for i in range(1,7)]].mean(axis=1)
            cc_df['avg_pay_amt'] = cc_df[[f'pay_amt{i}' for i in range(1,7)]].mean(axis=1)
            cc_df['max_pay_delay'] = cc_df[[f'pay_{i}' for i in [0,2,3,4,5,6]]].max(axis=1)
            cc_df['num_late_payments'] = cc_df[[f'pay_{i}' for i in [0,2,3,4,5,6]]].apply(lambda row: (row > 0).sum(), axis=1)
            cc_df['avg_pay_delay'] = cc_df[[f'pay_{i}' for i in [0,2,3,4,5,6]]].mean(axis=1)
            cc_df['was_ever_late'] = (cc_df[[f'pay_{i}' for i in [0,2,3,4,5,6]]] > 0).any(axis=1).astype(int)

            # Drop original columns
            bill_cols_to_drop = [f'bill_amt{i}' for i in range(1, 7)]
            pay_cols_to_drop = [f'pay_amt{i}' for i in range(1, 7)]
            pay2_cols_to_drop = [f'pay_{i}' for i in [0,2,3,4,5,6]]
            cols_to_drop = bill_cols_to_drop + pay_cols_to_drop + pay2_cols_to_drop
            cc_df.drop(columns=cols_to_drop, axis=1, inplace=True)
            cc_df.drop(columns=['id'], axis=1, inplace=True, errors='ignore')  # Drop 'id' if it exists
            
            return cc_df
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
            return None

    def make_predictions(model_data, X_data):
        """Make predictions using the loaded model pipeline (WITHOUT PCA)"""
        try:
            # Extract components
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Apply preprocessing pipeline (only scaling, no PCA)
            X_scaled = scaler.transform(X_data)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
            
            return y_pred, y_pred_proba
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None, None

    def plot_confusion_matrix(y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)
        plt.close()

    def plot_feature_importance(model_data, model_name):
        """Plot feature importance for tree-based models"""
        try:
            model = model_data['model']
            feature_names = model_data.get('feature_names', None)
            
            if hasattr(model, 'feature_importances_'):
                # Get feature importances
                importances = model.feature_importances_
                
                # Use provided feature names or create generic ones
                if feature_names is None:
                    feature_names = [f'Feature_{i+1}' for i in range(len(importances))]
                
                # Create a DataFrame for better handling
                feature_imp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Plot top 15 features
                top_features = feature_imp_df.head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
                plt.title(f'Top 15 Feature Importances - {model_name}')
                plt.xlabel('Importance')
                plt.ylabel('Features')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Display feature importance table
                st.subheader("Feature Importance Table")
                st.dataframe(feature_imp_df, use_container_width=True)
            else:
                st.warning(f"Feature importance not available for {model_name}")
        except Exception as e:
            st.error(f"Error plotting feature importance: {str(e)}")

    def plot_pr_curve(y_test, y_pred_proba):
        """Plot Precision-Recall curve"""
        try:
            # Calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
            average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, color='blue', lw=2, 
                   label=f'PR Curve (AP = {average_precision:.3f})')
            ax.fill_between(recall, precision, alpha=0.2, color='blue')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            return average_precision
        except Exception as e:
            st.error(f"Error plotting PR curve: {str(e)}")
            return None

    def plot_roc_curve(y_test, y_pred_proba):
        """Plot ROC curve"""
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random Classifier')
            ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            return roc_auc
        except Exception as e:
            st.error(f"Error plotting ROC curve: {str(e)}")
            return None

    def display_metrics(y_test, y_pred, selected_metrics):
        """Display selected metrics"""
        metrics_results = {}
        
        if 'Precision Score' in selected_metrics:
            precision = precision_score(y_test, y_pred)
            metrics_results['Precision'] = precision
            
        if 'Recall Score' in selected_metrics:
            recall = recall_score(y_test, y_pred)
            metrics_results['Recall'] = recall
            
        if 'F1 Score' in selected_metrics:
            f1 = f1_score(y_test, y_pred)
            metrics_results['F1 Score'] = f1
            
        if 'Accuracy Score' in selected_metrics:
            accuracy = accuracy_score(y_test, y_pred)
            metrics_results['Accuracy'] = accuracy
        
        # Display metrics in columns
        if metrics_results:
            st.subheader("Model Performance Metrics")
            cols = st.columns(len(metrics_results))
            for i, (metric, value) in enumerate(metrics_results.items()):
                with cols[i]:
                    st.metric(label=metric, value=f"{value:.4f}")

    def create_prediction_results_table(X_test_original, y_test, y_pred, y_pred_proba=None):
        """Create a comprehensive prediction results table"""
        # Create a copy of the test data
        results_df = X_test_original.copy()
        
        # Add actual and predicted values
        if y_test is not None:
            results_df['Actual_Default'] = y_test.values if hasattr(y_test, 'values') else y_test
            results_df['Actual_Default_Label'] = pd.Series(results_df['Actual_Default']).map({0: 'No Default', 1: 'Default'})
            # Add prediction accuracy for each row
            results_df['Prediction_Correct'] = (results_df['Actual_Default'] == y_pred)
        
        results_df['Predicted_Default'] = y_pred
        results_df['Predicted_Default_Label'] = pd.Series(y_pred).map({0: 'No Default', 1: 'Default'})
        
        # Add prediction probability if available
        if y_pred_proba is not None:
            results_df['Default_Probability'] = y_pred_proba[:, 1]  # Probability of default (class 1)
            results_df['Confidence'] = np.maximum(y_pred_proba[:, 0], y_pred_proba[:, 1])  # Max probability
        
        # Reorder columns to show predictions first
        prediction_cols = ['Predicted_Default_Label']
        if y_test is not None:
            prediction_cols = ['Actual_Default_Label', 'Predicted_Default_Label', 'Prediction_Correct']
        if y_pred_proba is not None:
            prediction_cols.insert(-1 if y_test is not None else 0, 'Default_Probability')
            prediction_cols.insert(-1 if y_test is not None else 1, 'Confidence')
        
        # Get original feature columns
        feature_cols = [col for col in results_df.columns if col not in prediction_cols + 
                       (['Actual_Default', 'Predicted_Default'] if y_test is not None else ['Predicted_Default'])]
        
        # Reorder dataframe
        results_df = results_df[prediction_cols + feature_cols]
        
        return results_df

    def display_classification_report(y_test, y_pred):
        """Display detailed classification report"""
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        
        st.subheader("Detailed Classification Report")
        st.dataframe(report_df.round(4), use_container_width=True)

    # Main app logic
    if uploaded_file is not None:
        # Load and process data
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            # Show basic info about the dataset
            st.info(f"Dataset shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
            
            # Apply feature engineering
            processed_df = feature_engineering(raw_df)
            
            if processed_df is not None:
                # Show processed data option
                if st.sidebar.checkbox("Show Processed Data", False):
                    st.subheader('Credit Card Default Dataset - After Feature Engineering')
                    st.write(processed_df)
                    st.info(f"Processed dataset shape: {processed_df.shape[0]} rows × {processed_df.shape[1]} columns")

                # Check if we have target variable for evaluation
                has_target = 'default payment next month' in processed_df.columns
                
                if has_target:
                    # Separate features and target
                    X = processed_df.drop('default payment next month', axis=1)
                    y = processed_df['default payment next month']
                    st.info("Dataset contains target variable - evaluation mode enabled")
                    
                    # Show class distribution
                    class_dist = y.value_counts()
                    st.sidebar.markdown("### Target Variable Distribution")
                    st.sidebar.write(f"No Default (0): {class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%)")
                    st.sidebar.write(f"Default (1): {class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%)")
                else:
                    # Only features available - prediction only mode
                    X = processed_df
                    y = None
                    st.info("Dataset contains only features - prediction only mode")

                # Model selection
                st.sidebar.subheader("Choose a Model")
                selected_model = st.sidebar.selectbox("Select Model", list(available_models.keys()))
                
                # Metrics selection (only if we have target variable)
                if has_target:
                    st.sidebar.subheader("Select Metrics to Display")
                    available_metrics = ["Confusion Matrix", "ROC Curve", "PR Curve", "Classification Report", 
                                       "Precision Score", "Recall Score", "F1 Score", "Accuracy Score"]
                    selected_metrics = st.sidebar.multiselect("Choose Metrics", available_metrics, 
                                                            default=["F1 Score", "Precision Score", "ROC Curve", "Confusion Matrix"])
                
                # Feature importance option
                show_feature_importance = st.sidebar.checkbox("Show Feature Importance", True)
                
                # Show prediction results option
                show_predictions = st.sidebar.checkbox("Show Prediction Results Table", True)
                
                # Prediction confidence threshold (only for prediction mode without target)
                if not has_target:
                    confidence_threshold = st.sidebar.slider(
                        "Confidence Threshold", 
                        min_value=0.5, 
                        max_value=1.0, 
                        value=0.8, 
                        step=0.05,
                        help="Only show predictions with confidence above this threshold"
                    )
                
                if st.sidebar.button("Make Predictions", key="make_predictions", type="primary"):
                    # Load the selected model
                    model_path = available_models[selected_model]
                    model_data = load_model(model_path)
                    
                    if model_data is not None:
                        st.subheader(f"{selected_model} Results")
                        
                        # Make predictions
                        with st.spinner('Making predictions...'):
                            y_pred, y_pred_proba = make_predictions(model_data, X)
                        
                        if y_pred is not None:
                            # Display basic prediction summary
                            total_predictions = len(y_pred)
                            default_predictions = (y_pred == 1).sum()
                            no_default_predictions = (y_pred == 0).sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Predictions", total_predictions)
                            with col2:
                                st.metric("Predicted Defaults", default_predictions)
                            with col3:
                                st.metric("Predicted No Defaults", no_default_predictions)
                            
                            # Display metrics only if we have target variable
                            if has_target and selected_metrics:
                                display_metrics(y, y_pred, selected_metrics)
                                
                                # Display classification report if selected
                                if 'Classification Report' in selected_metrics:
                                    display_classification_report(y, y_pred)
                                
                                # Display confusion matrix if selected
                                if 'Confusion Matrix' in selected_metrics:
                                    st.subheader("Confusion Matrix")
                                    plot_confusion_matrix(y, y_pred)
                                
                                # Display ROC curve if selected
                                if 'ROC Curve' in selected_metrics and y_pred_proba is not None:
                                    st.subheader("ROC Curve")
                                    roc_auc_score = plot_roc_curve(y, y_pred_proba)
                                    if roc_auc_score:
                                        st.info(f"ROC AUC Score: {roc_auc_score:.4f}")
                                
                                # Display PR curve if selected
                                if 'PR Curve' in selected_metrics and y_pred_proba is not None:
                                    st.subheader("Precision-Recall Curve")
                                    avg_precision = plot_pr_curve(y, y_pred_proba)
                                    if avg_precision:
                                        st.info(f"Average Precision Score: {avg_precision:.4f}")
                            
                            # Display feature importance
                            if show_feature_importance:
                                st.subheader("Feature Importance Analysis")
                                plot_feature_importance(model_data, selected_model)
                            
                            # Display prediction results table
                            if show_predictions:
                                st.subheader("Prediction Results")
                                
                                # Create and display the results table
                                results_df = create_prediction_results_table(X, y, y_pred, y_pred_proba)
                                
                                # Display summary statistics
                                if has_target:
                                    correct_predictions = results_df['Prediction_Correct'].sum()
                                    accuracy_pct = (correct_predictions / total_predictions) * 100
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Correct Predictions", correct_predictions)
                                    with col2:
                                        st.metric("Accuracy", f"{accuracy_pct:.2f}%")
                                    with col3:
                                        if y_pred_proba is not None:
                                            avg_confidence = results_df['Confidence'].mean()
                                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                                
                                # Add filters for the results table
                                st.subheader("Filter Results")
                                filter_col1, filter_col2 = st.columns(2)
                                
                                if has_target:
                                    with filter_col1:
                                        filter_actual = st.selectbox(
                                            "Filter by Actual Result:",
                                            ["All", "Default", "No Default"],
                                            key="filter_actual"
                                        )
                                    
                                    with filter_col2:
                                        filter_correct = st.selectbox(
                                            "Filter by Prediction Accuracy:",
                                            ["All", "Correct", "Incorrect"],
                                            key="filter_correct"
                                        )
                                else:
                                    with filter_col1:
                                        filter_pred = st.selectbox(
                                            "Filter by Prediction:",
                                            ["All", "Default", "No Default"],
                                            key="filter_pred"
                                        )
                                    
                                    with filter_col2:
                                        if y_pred_proba is not None:
                                            filter_confidence = st.selectbox(
                                                "Filter by Confidence:",
                                                ["All", f"High (>{confidence_threshold})", f"Low (<={confidence_threshold})"],
                                                key="filter_confidence"
                                            )
                                
                                # Apply filters
                                filtered_df = results_df.copy()
                                
                                if has_target:
                                    if filter_actual != "All":
                                        filtered_df = filtered_df[filtered_df['Actual_Default_Label'] == filter_actual]
                                    
                                    if filter_correct == "Correct":
                                        filtered_df = filtered_df[filtered_df['Prediction_Correct'] == True]
                                    elif filter_correct == "Incorrect":
                                        filtered_df = filtered_df[filtered_df['Prediction_Correct'] == False]
                                else:
                                    if filter_pred != "All":
                                        filtered_df = filtered_df[filtered_df['Predicted_Default_Label'] == filter_pred]
                                    
                                    if y_pred_proba is not None and 'filter_confidence' in locals():
                                        if filter_confidence.startswith("High"):
                                            filtered_df = filtered_df[filtered_df['Confidence'] > confidence_threshold]
                                        elif filter_confidence.startswith("Low"):
                                            filtered_df = filtered_df[filtered_df['Confidence'] <= confidence_threshold]
                                
                                # Display the filtered table
                                st.info(f"Showing {len(filtered_df)} of {len(results_df)} predictions")
                                st.dataframe(
                                    filtered_df, 
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Default_Probability": st.column_config.ProgressColumn(
                                            "Default Probability",
                                            help="Probability of default (0-1)",
                                            min_value=0,
                                            max_value=1,
                                            format="%.3f"
                                        ) if y_pred_proba is not None else None,
                                        "Confidence": st.column_config.ProgressColumn(
                                            "Confidence",
                                            help="Model confidence in prediction (0-1)",
                                            min_value=0,
                                            max_value=1,
                                            format="%.3f"
                                        ) if y_pred_proba is not None else None,
                                        "Prediction_Correct": st.column_config.CheckboxColumn(
                                            "Correct Prediction",
                                            help="Whether the prediction matches the actual result"
                                        ) if has_target else None
                                    }
                                )
                                
                                # Download button for results
                                csv = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Prediction Results as CSV",
                                    data=csv,
                                    file_name=f'{selected_model.lower().replace(" ", "_")}_predictions.csv',
                                    mime='text/csv'
                                )
                        else:
                            st.error("Failed to make predictions. Please check your model and data.")
                    else:
                        st.error("Failed to load the selected model.")
            else:
                st.error("Failed to process the dataset.")
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.error("Please make sure your CSV file has the correct format and column names.")
    else:
        st.info("Please upload a CSV file to get started.")
        st.markdown("""
        ### Instructions:
        1. **Upload your dataset** - CSV file with credit card customer data
        2. **Select a model** from the available trained models
        3. **Choose metrics** to display (if your dataset has target labels)
        4. **Click 'Make Predictions'** to see results and analysis
        5. **Use filters** to refine the results table
        6. **Download results** as CSV for further analysis
        
        ### Expected Dataset Format:
        Your CSV should contain columns like:
        - Customer demographic data (age, sex, education, marriage)
        - Credit data (limit_bal)
        - Payment history (pay_0, pay_2, pay_3, pay_4, pay_5, pay_6)
        - Bill amounts (bill_amt1 through bill_amt6)
        - Payment amounts (pay_amt1 through pay_amt6)
        - Target variable: 'default payment next month' (optional, for evaluation)
        """)

    # Contact section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Contact")
    st.sidebar.markdown("For questions or feedback, contact: **nonso.analytics@gmail.com**")

if __name__ == "__main__":
    main()