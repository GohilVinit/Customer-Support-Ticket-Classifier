# Customer Support Ticket Classification System
# Task 1 - Vijayi WFH Technologies Assignment

import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import pickle
import json

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    print("Some NLTK downloads failed. Please ensure internet connection.")

class TicketClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.issue_type_model = None
        self.urgency_model = None
        self.tfidf_vectorizer = None
        self.product_keywords = [
            'laptop', 'computer', 'phone', 'mobile', 'tablet', 'software', 
            'app', 'application', 'website', 'printer', 'monitor', 'keyboard',
            'mouse', 'headphones', 'camera', 'router', 'wifi', 'internet'
        ]
        self.complaint_keywords = [
            'broken', 'damaged', 'not working', 'error', 'bug', 'crash',
            'slow', 'late', 'delayed', 'missing', 'lost', 'wrong', 'incorrect',
            'failed', 'problem', 'issue', 'trouble', 'defective', 'faulty'
        ]
        
    def load_data(self, file_path):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Please check the file path.")
            return None
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None
    
    def explore_data(self, df):
        """Explore the dataset"""
        print("=== DATA EXPLORATION ===")
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Display sample data
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Distribution of target variables
        if 'issue_type' in df.columns:
            print(f"\nIssue Type distribution:")
            print(df['issue_type'].value_counts())
            
        if 'urgency_level' in df.columns:
            print(f"\nUrgency Level distribution:")
            print(df['urgency_level'].value_counts())
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        if pd.isna(text) or text == "":
            return ""
        
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, df):
        """Extract additional features from text"""
        features_df = df.copy()
        
        # Text length features
        features_df['text_length'] = df['ticket_text'].str.len()
        features_df['word_count'] = df['ticket_text'].str.split().str.len()
        
        # Sentiment analysis
        features_df['sentiment_polarity'] = df['ticket_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
        features_df['sentiment_subjectivity'] = df['ticket_text'].apply(
            lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
        )
        
        # Count of complaint keywords
        features_df['complaint_keyword_count'] = df['ticket_text'].apply(
            lambda x: sum(1 for keyword in self.complaint_keywords 
                         if keyword in str(x).lower()) if pd.notna(x) else 0
        )
        
        # Urgency indicators (keywords that might indicate urgency)
        urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        features_df['urgency_keyword_count'] = df['ticket_text'].apply(
            lambda x: sum(1 for keyword in urgency_keywords 
                         if keyword in str(x).lower()) if pd.notna(x) else 0
        )
        
        return features_df
    
    def prepare_data(self, df):
        """Comprehensive data preparation"""
        print("=== DATA PREPARATION ===")
        
        # Handle missing values more carefully
        df = df.copy()
        
        # Fill missing ticket_text with empty string
        df['ticket_text'] = df['ticket_text'].fillna('')
        
        # Handle missing issue_type - use 'General Inquiry' as default
        if 'issue_type' in df.columns:
            df['issue_type'] = df['issue_type'].fillna('General Inquiry')
        
        # Handle missing urgency_level - use 'Medium' as default  
        if 'urgency_level' in df.columns:
            df['urgency_level'] = df['urgency_level'].fillna('Medium')
        
        # Handle missing product
        if 'product' in df.columns:
            df['product'] = df['product'].fillna('Unknown Product')
        
        # Remove rows with completely empty ticket_text
        df = df[df['ticket_text'].str.strip() != '']
        
        # Clean and preprocess text
        print("Preprocessing text...")
        df['cleaned_text'] = df['ticket_text'].apply(self.preprocess_text)
        
        # Remove rows where cleaned text is empty
        df = df[df['cleaned_text'].str.strip() != '']
        
        # Extract additional features
        print("Extracting features...")
        df = self.extract_features(df)
        
        print(f"Data preparation completed. Final shape: {df.shape}")
        return df
    
    def extract_entities(self, text):
        """Extract entities from text using rule-based methods"""
        entities = {
            'products': [],
            'dates': [],
            'complaint_keywords': []
        }
        
        if pd.isna(text):
            return entities
        
        text_lower = str(text).lower()
        
        # Extract product names
        for product in self.product_keywords:
            if product in text_lower:
                entities['products'].append(product)
        
        # Extract dates using regex
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text)
            entities['dates'].extend(dates)
        
        # Extract complaint keywords
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                entities['complaint_keywords'].append(keyword)
        
        # Remove duplicates
        entities['products'] = list(set(entities['products']))
        entities['dates'] = list(set(entities['dates']))
        entities['complaint_keywords'] = list(set(entities['complaint_keywords']))
        
        return entities
    
    def train_models(self, df):
        """Train classification models"""
        print("=== MODEL TRAINING ===")
        
        # Check if we have enough data for training
        if len(df) < 10:
            print("Warning: Very small dataset. Results may not be reliable.")
        
        # Prepare features
        X_text = df['cleaned_text']
        additional_features = df[['text_length', 'word_count', 'sentiment_polarity', 
                                'sentiment_subjectivity', 'complaint_keyword_count', 
                                'urgency_keyword_count']].values
        
        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(5000, len(df) * 10),  # Adjust max_features based on dataset size
            ngram_range=(1, 2),
            min_df=1,  # Reduced for small datasets
            max_df=0.95
        )
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
        
        # Combine TF-IDF with additional features
        X_combined = np.hstack([X_tfidf.toarray(), additional_features])
        
        # Train Issue Type Classifier
        if 'issue_type' in df.columns:
            print("Training Issue Type Classifier...")
            y_issue = df['issue_type']
            
            # Check class distribution
            print(f"Issue type distribution:\n{y_issue.value_counts()}")
            
            # Determine if we can use stratified split
            min_class_count = y_issue.value_counts().min()
            use_stratify = min_class_count >= 2 and len(df) >= 10
            
            if use_stratify:
                # Split data with stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y_issue, test_size=0.2, random_state=42, stratify=y_issue
                )
                cv_folds = min(5, min_class_count)  # Adjust CV folds
            else:
                # Split data without stratification for small datasets
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y_issue, test_size=0.2, random_state=42
                )
                cv_folds = min(3, len(df) // 3)  # Fewer folds for small data
            
            # Try different models
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),  # Fewer trees for small data
            }
            
            # Add SVM only for larger datasets
            if len(df) >= 20:
                models['SVM'] = SVC(kernel='linear', random_state=42, probability=True)
            
            best_score = 0
            best_model = None
            
            for name, model in models.items():
                try:
                    if cv_folds >= 2:
                        scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                        print(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                        avg_score = scores.mean()
                    else:
                        # For very small datasets, just fit and score on training data
                        model.fit(X_train, y_train)
                        avg_score = model.score(X_train, y_train)
                        print(f"{name} Training Score: {avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            if best_model is not None:
                # Train best model
                best_model.fit(X_train, y_train)
                self.issue_type_model = best_model
                
                # Evaluate if we have test data
                if len(X_test) > 0:
                    y_pred = best_model.predict(X_test)
                    print(f"\nIssue Type Classifier Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred, zero_division=0))
            else:
                print("Could not train any issue type model successfully")
        
        # Train Urgency Level Classifier
        if 'urgency_level' in df.columns:
            print("\nTraining Urgency Level Classifier...")
            y_urgency = df['urgency_level']
            
            # Check class distribution
            print(f"Urgency level distribution:\n{y_urgency.value_counts()}")
            
            # Determine if we can use stratified split
            min_class_count = y_urgency.value_counts().min()
            use_stratify = min_class_count >= 2 and len(df) >= 10
            
            if use_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y_urgency, test_size=0.2, random_state=42, stratify=y_urgency
                )
                cv_folds = min(5, min_class_count)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y_urgency, test_size=0.2, random_state=42
                )
                cv_folds = min(3, len(df) // 3)
            
            best_score = 0
            best_model = None
            
            for name, model in models.items():
                try:
                    if cv_folds >= 2:
                        scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                        print(f"{name} CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                        avg_score = scores.mean()
                    else:
                        model.fit(X_train, y_train)
                        avg_score = model.score(X_train, y_train)
                        print(f"{name} Training Score: {avg_score:.4f}")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            if best_model is not None:
                # Train best model
                best_model.fit(X_train, y_train)
                self.urgency_model = best_model
                
                # Evaluate if we have test data
                if len(X_test) > 0:
                    y_pred = best_model.predict(X_test)
                    print(f"\nUrgency Level Classifier Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred, zero_division=0))
            else:
                print("Could not train any urgency level model successfully")
    
    def predict_ticket(self, ticket_text):
        """Main prediction function for a single ticket"""
        if not self.tfidf_vectorizer or not self.issue_type_model or not self.urgency_model:
            return {
                'error': 'Models not trained. Please train models first.',
                'predicted_issue_type': None,
                'predicted_urgency_level': None,
                'extracted_entities': {}
            }
        
        # Preprocess text
        cleaned_text = self.preprocess_text(ticket_text)
        
        # Extract additional features
        text_length = len(ticket_text)
        word_count = len(ticket_text.split())
        sentiment = TextBlob(ticket_text).sentiment
        sentiment_polarity = sentiment.polarity
        sentiment_subjectivity = sentiment.subjectivity
        
        complaint_keyword_count = sum(1 for keyword in self.complaint_keywords 
                                    if keyword in ticket_text.lower())
        
        urgency_keywords = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        urgency_keyword_count = sum(1 for keyword in urgency_keywords 
                                  if keyword in ticket_text.lower())
        
        additional_features = np.array([[text_length, word_count, sentiment_polarity, 
                                       sentiment_subjectivity, complaint_keyword_count, 
                                       urgency_keyword_count]])
        
        # Vectorize text
        X_tfidf = self.tfidf_vectorizer.transform([cleaned_text])
        
        # Combine features
        X_combined = np.hstack([X_tfidf.toarray(), additional_features])
        
        # Make predictions
        predicted_issue_type = self.issue_type_model.predict(X_combined)[0]
        predicted_urgency_level = self.urgency_model.predict(X_combined)[0]
        
        # Get confidence scores if available
        confidence_scores = {}
        try:
            if hasattr(self.issue_type_model, 'predict_proba'):
                confidence_scores['issue_type_proba'] = self.issue_type_model.predict_proba(X_combined)[0].max()
            else:
                confidence_scores['issue_type_proba'] = 1.0
        except:
            confidence_scores['issue_type_proba'] = 1.0
        
        try:
            if hasattr(self.urgency_model, 'predict_proba'):
                confidence_scores['urgency_proba'] = self.urgency_model.predict_proba(X_combined)[0].max()
            else:
                confidence_scores['urgency_proba'] = 1.0
        except:
            confidence_scores['urgency_proba'] = 1.0
        
        # Extract entities
        extracted_entities = self.extract_entities(ticket_text)
        
        return {
            'predicted_issue_type': predicted_issue_type,
            'predicted_urgency_level': predicted_urgency_level,
            'extracted_entities': extracted_entities,
            'confidence_scores': confidence_scores
        }
    
    def save_models(self, filepath_prefix='ticket_classifier'):
        """Save trained models"""
        try:
            with open(f'{filepath_prefix}_issue_model.pkl', 'wb') as f:
                pickle.dump(self.issue_type_model, f)
            
            with open(f'{filepath_prefix}_urgency_model.pkl', 'wb') as f:
                pickle.dump(self.urgency_model, f)
            
            with open(f'{filepath_prefix}_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            print(f"Models saved with prefix: {filepath_prefix}")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def load_models(self, filepath_prefix='ticket_classifier'):
        """Load trained models"""
        try:
            with open(f'{filepath_prefix}_issue_model.pkl', 'rb') as f:
                self.issue_type_model = pickle.load(f)
            
            with open(f'{filepath_prefix}_urgency_model.pkl', 'rb') as f:
                self.urgency_model = pickle.load(f)
            
            with open(f'{filepath_prefix}_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            print(f"Models loaded successfully from: {filepath_prefix}")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def visualize_data(self, df):
        """Create visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Issue type distribution
        if 'issue_type' in df.columns:
            df['issue_type'].value_counts().plot(kind='bar', ax=axes[0,0], rot=45)
            axes[0,0].set_title('Issue Type Distribution')
            axes[0,0].set_xlabel('Issue Type')
            axes[0,0].set_ylabel('Count')
        
        # Urgency level distribution
        if 'urgency_level' in df.columns:
            df['urgency_level'].value_counts().plot(kind='bar', ax=axes[0,1], rot=45)
            axes[0,1].set_title('Urgency Level Distribution')
            axes[0,1].set_xlabel('Urgency Level')
            axes[0,1].set_ylabel('Count')
        
        # Text length distribution
        if 'text_length' in df.columns:
            df['text_length'].hist(bins=30, ax=axes[1,0])
            axes[1,0].set_title('Text Length Distribution')
            axes[1,0].set_xlabel('Text Length')
            axes[1,0].set_ylabel('Frequency')
        
        # Sentiment distribution
        if 'sentiment_polarity' in df.columns:
            df['sentiment_polarity'].hist(bins=30, ax=axes[1,1])
            axes[1,1].set_title('Sentiment Polarity Distribution')
            axes[1,1].set_xlabel('Sentiment Polarity')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    # Initialize classifier
    classifier = TicketClassifier()
    
    # Load data (replace with your actual file path)
    file_path = r"C:\internshala\ticket_classifier\data\ai_dev_assignment_tickets_complex_1000.xls"
    print("Attempting to load data...")
    print(f"Expected file: {file_path}")
    print("Please ensure the Excel file is in the same directory or provide the full path.")
    
    df = classifier.load_data(file_path)
    
    if df is not None:
        # Explore data
        classifier.explore_data(df)
        
        # Prepare data
        df_processed = classifier.prepare_data(df)
        
        # Train models
        classifier.train_models(df_processed)
        
        # Save models
        classifier.save_models()
        
        # Create visualizations
        classifier.visualize_data(df_processed)
        
        # Test prediction with sample text
        sample_ticket = "My laptop is broken and not starting up. This is urgent as I have a presentation tomorrow."
        print("\n=== SAMPLE PREDICTION ===")
        print(f"Input: {sample_ticket}")
        result = classifier.predict_ticket(sample_ticket)
        print(f"Result: {json.dumps(result, indent=2)}")
        
    else:
        print("\nData file not found. Here's what you need to do:")
        print("1. Place the Excel file 'ai_dev_assignment_tickets_complex_1000.xlsx' in the same directory")
        print("2. Or modify the 'file_path' variable with the correct path to your file")
        print("3. Ensure the file has columns: ticket_id, ticket_text, issue_type, urgency_level, product")
        
        # Demo with synthetic data
        print("\nCreating demo with synthetic data...")
        demo_data = {
            'ticket_id': list(range(1, 21)),  # More data points
            'ticket_text': [
                "My laptop screen is broken and flickering constantly",
                "The software crashes every time I try to open it", 
                "I need help setting up my new phone",
                "The printer is not working and I need it urgently",
                "My internet connection is very slow",
                "Payment issue with my recent order, was charged twice",
                "Can you tell me about warranty for my tablet?",
                "Wrong item delivered, ordered phone got laptop instead",
                "Installation fails at step 2 for the new software",
                "My order is 5 days late, where is my package?",
                "The device stopped working after 3 days of use",
                "Need information about return policy",
                "Billing error on my account, wrong amount charged",
                "Setup assistance needed for smart TV",
                "Product defective, making strange noises",
                "General inquiry about product features",
                "Late delivery, ordered 2 weeks ago",
                "Installation problem with new router",
                "Defective headphones, no sound from left side",
                "Help needed with software configuration"
            ],
            'issue_type': [
                'Hardware', 'Software', 'Setup', 'Hardware', 'Network',
                'Billing Problem', 'General Inquiry', 'Wrong Item', 'Installation Issue', 'Late Delivery',
                'Product Defect', 'General Inquiry', 'Billing Problem', 'Setup', 'Product Defect',
                'General Inquiry', 'Late Delivery', 'Installation Issue', 'Product Defect', 'Setup'
            ],
            'urgency_level': [
                'High', 'Medium', 'Low', 'High', 'Medium',
                'Medium', 'Low', 'Medium', 'Low', 'Medium',
                'High', 'Low', 'Medium', 'Low', 'High',
                'Low', 'Medium', 'Low', 'High', 'Low'
            ],
            'product': [
                'laptop', 'software', 'phone', 'printer', 'internet',
                'payment', 'tablet', 'phone', 'software', 'package',
                'device', 'general', 'billing', 'smart TV', 'product',
                'general', 'delivery', 'router', 'headphones', 'software'
            ]
        }
        
        df_demo = pd.DataFrame(demo_data)
        print("Demo data created:")
        print(df_demo)
        
        # Process demo data
        df_processed = classifier.prepare_data(df_demo)
        classifier.train_models(df_processed)
        
        # Test prediction
        sample_ticket = "My computer mouse is not working properly and clicks are not registering"
        print(f"\nTesting with: {sample_ticket}")
        result = classifier.predict_ticket(sample_ticket)
        print(f"Prediction result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()