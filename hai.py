import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"C:\Users\lenovo\Desktop\lung cancer project\survey_lung_data.csv"
MODEL_PATH = os.path.join(CURRENT_DIR, 'lung_cancer_model.pkl')
SCALER_PATH = os.path.join(CURRENT_DIR, 'scaler.pkl')

class LungCancerPredictionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.feature_names = None
        
    def load_data(self):
        """Load and preprocess the lung cancer dataset"""
        try:
            print(f"Loading dataset from: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            
            # Display initial data info
            print("\nDataset Overview:")
            print(f"Total samples: {len(df)}")
            print("\nColumns in dataset:")
            print(df.columns.tolist())
            
            # Store feature names
            self.feature_names = [col for col in df.columns if col != 'LUNG_CANCER']
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            # Separate features and target
            X = df[self.feature_names]
            y = df['LUNG_CANCER']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Split the data with stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("\nData preprocessing completed successfully")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Testing samples: {len(self.X_test)}")
            
            # Store data information for GUI
            self.data = {
                'feature_names': self.feature_names,
                'categorical_columns': ['GENDER'],
                'numeric_columns': [col for col in self.feature_names if col != 'GENDER']
            }
            
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            raise

    def preprocess_data(self, df):
        """Preprocess the data"""
        data = df.copy()
        
        # Convert categorical variables
        data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
        data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        # Handle missing values
        data = data.fillna(data.median())
        
        return data
        
    def build_model(self):
        """Build and train the Random Forest model with grid search"""
        print("\nTraining Random Forest model with grid search...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        # Initialize base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Print results
        print("\nBest Parameters:", grid_search.best_params_)
        print(f"Cross-validation accuracy: {grid_search.best_score_:.3f}")
        
        # Calculate accuracies
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        
        print(f"\nFinal Training Accuracy: {train_accuracy:.3f}")
        print(f"Final Testing Accuracy: {test_accuracy:.3f}")
        
    def evaluate_model(self):
        """Evaluate the model and display results"""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Plot feature importance
        self.plot_feature_importance()
        
    def plot_feature_importance(self):
        """Plot feature importance scores"""
        importance = self.model.feature_importances_
        features_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=features_df)
        plt.title('Feature Importance in Lung Cancer Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def save_model(self):
        """Save the trained model and scaler"""
        try:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            
            # Save feature names and data information
            with open(os.path.join(CURRENT_DIR, 'model_info.pkl'), 'wb') as f:
                joblib.dump(self.data, f)
                
            print(f"\nModel and related files saved successfully at: {CURRENT_DIR}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

class PredictionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lung Cancer Risk Prediction System")
        self.root.geometry("800x900")
        
        try:
            # Load model and related files
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.data = joblib.load(os.path.join(CURRENT_DIR, 'model_info.pkl'))
                
            self.setup_gui()
            
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Please train the model first.")
            self.root.destroy()
            return
            
    def setup_gui(self):
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)
        
        # Add canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar components
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(
            scrollable_frame,
            text="Lung Cancer Risk Prediction",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=20)
        
        # Create input fields
        self.inputs = {}
        
        # Create sections for different types of inputs
        self.create_demographic_section(scrollable_frame)
        self.create_symptoms_section(scrollable_frame)
        
        # Predict button
        predict_button = ttk.Button(
            scrollable_frame,
            text="Predict Risk",
            command=self.predict,
            style='Accent.TButton'
        )
        predict_button.pack(pady=20)
        
        # Result section
        self.create_result_section(scrollable_frame)
        
        # Configure style
        self.configure_styles()
        
    def create_demographic_section(self, parent):
        """Create demographic information section"""
        frame = ttk.LabelFrame(parent, text="Demographic Information", padding=10)
        frame.pack(fill="x", padx=20, pady=10)
        
        # Age input
        age_frame = ttk.Frame(frame)
        age_frame.pack(fill="x", pady=5)
        ttk.Label(age_frame, text="AGE:").pack(side=tk.LEFT)
        self.inputs['AGE'] = tk.StringVar(value="0")
        age_spinbox = ttk.Spinbox(
            age_frame,
            from_=0,
            to=120,
            textvariable=self.inputs['AGE'],
            width=10
        )
        age_spinbox.pack(side=tk.LEFT, padx=10)
        
        # Gender input
        gender_frame = ttk.Frame(frame)
        gender_frame.pack(fill="x", pady=5)
        ttk.Label(gender_frame, text="GENDER:").pack(side=tk.LEFT)
        self.inputs['GENDER'] = tk.StringVar(value="M")
        ttk.Radiobutton(gender_frame, text="Male", variable=self.inputs['GENDER'], value="M").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(gender_frame, text="Female", variable=self.inputs['GENDER'], value="F").pack(side=tk.LEFT)
        
    def create_symptoms_section(self, parent):
        """Create symptoms section"""
        frame = ttk.LabelFrame(parent, text="Symptoms & Risk Factors", padding=10)
        frame.pack(fill="x", padx=20, pady=10)
        
        # Create binary inputs for all other features
        for feature in self.data['feature_names']:
            if feature not in ['AGE', 'GENDER']:
                self.create_binary_input(frame, feature)
                
    def create_binary_input(self, parent, feature):
        """Create a binary input field"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        # Convert feature name to display text
        display_name = feature.replace('_', ' ').title()
        
        ttk.Label(frame, text=f"{display_name}:").pack(side=tk.LEFT)
        self.inputs[feature] = tk.StringVar(value="0")
        ttk.Radiobutton(frame, text="No", variable=self.inputs[feature], value="0").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(frame, text="Yes", variable=self.inputs[feature], value="1").pack(side=tk.LEFT)
        
    def create_result_section(self, parent):
        """Create result section"""
        frame = ttk.LabelFrame(parent, text="Prediction Result", padding=10)
        frame.pack(fill="x", padx=20, pady=10)
        
        self.result_label = ttk.Label(
            frame,
            text="Enter patient information and click Predict",
            font=('Arial', 12)
        )
        self.result_label.pack(pady=10)
        
        # Risk meter
        self.risk_canvas = tk.Canvas(
            frame,
            width=300,
            height=30,
            bg='white',
            bd=1,
            relief='solid'
        )
        self.risk_canvas.pack(pady=10)
        
    def configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 12))
        style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
        
    def update_risk_meter(self, probability):
        """Update the risk meter visualization"""
        self.risk_canvas.delete('all')
        
        # Create gradient
        width = int(300 * probability)
        if probability < 0.4:
            color = 'green'
        elif probability < 0.7:
            color = 'orange'
        else:
            color = 'red'
            
        self.risk_canvas.create_rectangle(0, 0, width, 30, fill=color)
        self.risk_canvas.create_text(
            150, 15,
            text=f"{probability*100:.1f}% Risk",
            fill='black'
        )
        
    def predict(self):
        """Make prediction based on input values"""
        try:
            # Gather input values
            input_data = {}
            for name, var in self.inputs.items():
                value = var.get()
                if name == 'GENDER':
                    value = 1 if value == 'M' else 0
                input_data[name] = float(value)
            
            # Create feature vector
            feature_vector = pd.DataFrame([input_data])
            
            # Ensure correct column order
            feature_vector = feature_vector[self.data['feature_names']]
            
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)
            
            # Make prediction
            probability = self.model.predict_proba(scaled_features)[0][1]
            prediction = "High Risk" if probability >= 0.5 else "Low Risk"
            
            # Update GUI
            self.result_label.config(
                text=f"Prediction: {prediction}",
                foreground='red' if prediction == "High Risk" else 'green'
            )
            self.update_risk_meter(probability)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}\nPlease check all inputs.")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    try:
        # Train the model
        print("Initializing Lung Cancer Prediction System...")
        system = LungCancerPredictionSystem()
        
        print("\nLoading and preprocessing data...")
        system.load_data()
        
        print("\nBuilding and training model...")
        system.build_model()
        
        print("\nEvaluating model performance...")
        system.evaluate_model()
        
        print("\nSaving model and related files...")
        system.save_model()
        
        print("\nLaunching prediction interface...")
        app = PredictionApp()
        app.run()
        
    except FileNotFoundError as e:
        print(f"\nError: Dataset not found. Please ensure 'survey_lung_data.csv' is in the correct location.")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()