# Gradio Web Interface for Customer Support Ticket Classification
# This file should be run after training the models in the main script

import gradio as gr
import json
import pandas as pd
from ticket_classifier import TicketClassifier  # Import the main class

class GradioTicketApp:
    def __init__(self, model_prefix='ticket_classifier'):
        self.classifier = TicketClassifier()
        self.model_prefix = model_prefix
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.classifier.load_models(self.model_prefix)
            self.models_loaded = True
            print("Models loaded successfully for Gradio app")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Please train models first by running the main script")
            self.models_loaded = False
    
    def predict_single_ticket(self, ticket_text):
        """Predict for a single ticket and format output"""
        if not self.models_loaded:
            return "❌ Models not loaded. Please train models first.", "", ""
        
        if not ticket_text.strip():
            return "❌ Please enter a ticket description.", "", ""
        
        try:
            # Get prediction
            result = self.classifier.predict_ticket(ticket_text)
            
            # Format the output
            issue_type = result.get('predicted_issue_type', 'Unknown')
            urgency_level = result.get('predicted_urgency_level', 'Unknown')
            entities = result.get('extracted_entities', {})
            confidence = result.get('confidence_scores', {})
            
            # Create formatted output strings
            prediction_output = f"""
🎯 **Issue Type**: {issue_type}
⚡ **Urgency Level**: {urgency_level}

📊 **Confidence Scores**:
- Issue Type: {confidence.get('issue_type_proba', 0):.2%}
- Urgency Level: {confidence.get('urgency_proba', 0):.2%}
            """
            
            entities_output = f"""
🛍️ **Products Found**: {', '.join(entities.get('products', [])) or 'None'}
📅 **Dates Found**: {', '.join(entities.get('dates', [])) or 'None'}
⚠️ **Complaint Keywords**: {', '.join(entities.get('complaint_keywords', [])) or 'None'}
            """
            
            json_output = json.dumps(result, indent=2)
            
            return prediction_output, entities_output, json_output
            
        except Exception as e:
            error_msg = f"❌ Error during prediction: {str(e)}"
            return error_msg, "", ""
    
    def predict_batch_tickets(self, csv_file):
        """Process multiple tickets from CSV file"""
        if not self.models_loaded:
            return "❌ Models not loaded. Please train models first."
        
        if csv_file is None:
            return "❌ Please upload a CSV file."
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file.name)
            
            if 'ticket_text' not in df.columns:
                return "❌ CSV file must contain 'ticket_text' column."
            
            # Process each ticket
            results = []
            for idx, row in df.iterrows():
                ticket_text = row['ticket_text']
                result = self.classifier.predict_ticket(ticket_text)
                
                # Flatten the result for CSV output
                flat_result = {
                    'ticket_id': row.get('ticket_id', idx),
                    'ticket_text': ticket_text,
                    'predicted_issue_type': result.get('predicted_issue_type'),
                    'predicted_urgency_level': result.get('predicted_urgency_level'),
                    'products_found': ', '.join(result.get('extracted_entities', {}).get('products', [])),
                    'dates_found': ', '.join(result.get('extracted_entities', {}).get('dates', [])),
                    'complaint_keywords': ', '.join(result.get('extracted_entities', {}).get('complaint_keywords', [])),
                    'issue_confidence': result.get('confidence_scores', {}).get('issue_type_proba', 0),
                    'urgency_confidence': result.get('confidence_scores', {}).get('urgency_proba', 0)
                }
                results.append(flat_result)
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save to CSV
            output_file = 'batch_predictions.csv'
            results_df.to_csv(output_file, index=False)
            
            summary = f"""
✅ **Batch Processing Complete**
- Processed {len(results)} tickets
- Results saved to: {output_file}

📊 **Summary**:
- Issue Types: {results_df['predicted_issue_type'].value_counts().to_dict()}
- Urgency Levels: {results_df['predicted_urgency_level'].value_counts().to_dict()}
            """
            
            return summary
            
        except Exception as e:
            return f"❌ Error processing batch: {str(e)}"
    
    def create_interface(self):
        """Create and return Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        """
        
        with gr.Blocks(css=custom_css, title="Customer Support Ticket Classifier") as interface:
            
            gr.Markdown("""
            # 🎫 Customer Support Ticket Classifier
            
            This tool classifies customer support tickets by **issue type** and **urgency level**, 
            and extracts key entities like products, dates, and complaint keywords.
            
            Choose between single ticket prediction or batch processing multiple tickets.
            """)
            
            with gr.Tabs():
                # Single Ticket Tab
                with gr.TabItem("🔍 Single Ticket Analysis"):
                    gr.Markdown("### Analyze a single customer support ticket")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            ticket_input = gr.Textbox(
                                label="Customer Ticket Description",
                                placeholder="Enter the customer's issue description here...",
                                lines=5,
                                max_lines=10
                            )
                            
                            predict_btn = gr.Button(
                                "🔍 Analyze Ticket", 
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=3):
                            prediction_output = gr.Markdown(
                                label="Prediction Results",
                                value="Results will appear here..."
                            )
                            
                            entities_output = gr.Markdown(
                                label="Extracted Entities",
                                value="Extracted entities will appear here..."
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📄 JSON Output")
                            json_output = gr.Code(
                                label="Raw JSON Response",
                                language="json",
                                lines=10
                            )
                    
                    # Examples
                    gr.Markdown("### 📝 Example Tickets (Click to try)")
                    example_tickets = [
                        ["My laptop screen is cracked and I can't see anything clearly. This needs to be fixed urgently as I have work tomorrow."],
                        ["The mobile app keeps crashing when I try to login. Started happening after the latest update."],
                        ["I need help setting up my new printer with the wifi network. Not urgent but would appreciate guidance."],
                        ["The website is loading very slowly and sometimes doesn't load at all. This is affecting my daily work."],
                        ["My headphones have a broken wire and the left ear stopped working completely."]
                    ]
                    
                    gr.Examples(
                        examples=example_tickets,
                        inputs=[ticket_input],
                        label="Try these examples"
                    )
                
                # Batch Processing Tab
                with gr.TabItem("📊 Batch Processing"):
                    gr.Markdown("### Process multiple tickets from a CSV file")
                    gr.Markdown("""
                    **CSV Format Requirements:**
                    - Must contain a `ticket_text` column with the ticket descriptions
                    - Optional: `ticket_id` column for identification
                    - Other columns will be preserved in the output
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            csv_input = gr.File(
                                label="Upload CSV File",
                                file_types=[".csv"],
                                type="filepath"
                            )
                            
                            batch_btn = gr.Button(
                                "📊 Process Batch", 
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column():
                            batch_output = gr.Markdown(
                                label="Batch Processing Results",
                                value="Upload a CSV file and click 'Process Batch' to see results..."
                            )
                
                # Model Information Tab
                with gr.TabItem("ℹ️ Model Information"):
                    gr.Markdown("""
                    ### 🤖 Model Architecture & Features
                    
                    **Classification Models:**
                    - **Issue Type Classifier**: Multi-class classification for categorizing ticket types
                    - **Urgency Level Classifier**: Predicts Low, Medium, or High urgency
                    
                    **Feature Engineering:**
                    - **TF-IDF Vectorization**: Captures word importance and n-grams (1-2)
                    - **Text Statistics**: Length, word count, sentiment analysis
                    - **Keyword Matching**: Complaint and urgency keyword detection
                    
                    **Entity Extraction:**
                    - **Products**: Hardware/software items mentioned
                    - **Dates**: Temporal information extraction
                    - **Complaint Keywords**: Issue-indicating terms
                    
                    **Model Selection:**
                    - Cross-validation used to select best performing model
                    - Options: Logistic Regression, Random Forest, SVM
                    - Performance metrics: Accuracy, Precision, Recall, F1-score
                    """)
                    
                    if self.models_loaded:
                        gr.Markdown("✅ **Status**: Models loaded and ready")
                    else:
                        gr.Markdown("❌ **Status**: Models not loaded - please train first")
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_single_ticket,
                inputs=[ticket_input],
                outputs=[prediction_output, entities_output, json_output]
            )
            
            batch_btn.click(
                fn=self.predict_batch_tickets,
                inputs=[csv_input],
                outputs=[batch_output]
            )
        
        return interface

def launch_gradio_app():
    """Launch the Gradio application"""
    app = GradioTicketApp()
    interface = app.create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed error messages
    )

if __name__ == "__main__":
    launch_gradio_app()