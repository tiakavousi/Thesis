import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, roc_curve, auc
)
import logging
import pandas as pd
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display, HTML

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Handles model evaluation and generates comprehensive PDF reports.
    """

    def __init__(self, model, device, model_name, class_names):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.class_names = class_names
        
        # Enhanced color palette for visualizations
        self.color_palette = sns.color_palette("viridis", len(class_names) if class_names else 2)

    def evaluate(self, dataloader):
        """Evaluate the model and return labels, predictions, and probabilities"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # Store probabilities for ROC curves
        
        print(f"[INFO] Evaluating model on {len(dataloader)} batches...")

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get raw logits and convert to probabilities
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                predictions = torch.argmax(logits, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        print(f"[INFO] Evaluation complete. Processed {len(all_labels)} samples.")
        return all_labels, all_preds, np.array(all_probs)

    def calculate_metrics(self, all_labels, all_preds, all_probs=None):
        """
        Calculate and return comprehensive evaluation metrics.
        
        Args:
            all_labels: List of true labels
            all_preds: List of predicted labels
            all_probs: Array of prediction probabilities (optional)
            
        Returns:
            Dict containing various metrics
        """
        print("[INFO] Calculating metrics...")
        
        # Determine if binary or multiclass
        unique_labels = sorted(list(set(all_labels)))
        is_binary = len(unique_labels) == 2
        
        # Handle average method based on problem type
        if is_binary:
            average_method = 'binary'
        else:
            average_method = 'macro'
            
        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average=average_method, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=average_method, zero_division=0),
            "f1": f1_score(all_labels, all_preds, average=average_method, zero_division=0)
        }
        
        # Get detailed classification report
        report = classification_report(all_labels, all_preds, 
                                     target_names=self.class_names if self.class_names else None,
                                     output_dict=True)
        
        # Store per-class metrics
        metrics["per_class"] = {k: v for k, v in report.items() 
                              if k not in ['accuracy', 'macro avg', 'weighted avg']}
        
        # Store the confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
        
        # For binary classification, add ROC and precision-recall metrics
        if all_probs is not None and is_binary:
            # For binary classification, we need the probability of the positive class
            pos_probs = all_probs[:, 1]
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(all_labels, pos_probs)
            metrics["roc_auc"] = auc(fpr, tpr)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
            
            # Calculate precision-recall curve and average precision
            precision, recall, _ = precision_recall_curve(all_labels, pos_probs)
            metrics["average_precision"] = average_precision_score(all_labels, pos_probs)
            metrics["pr_curve"] = {"precision": precision, "recall": recall}
            
        print(f"[INFO] Metrics calculated. Overall accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def _create_confusion_matrix_fig(self, confusion_mat, normalize=False):
        """Create and return a confusion matrix figure with improved visualization."""
        # Normalize if requested
        if normalize:
            confusion_mat_display = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title_suffix = " (Normalized)"
        else:
            confusion_mat_display = confusion_mat
            fmt = 'd'
            title_suffix = ""
        
        # Get class labels
        labels = self.class_names if self.class_names else [str(i) for i in range(len(confusion_mat))]
        
        # Create a figure
        plt.figure(figsize=(10, 8))
        
        # Use a better colormap for the confusion matrix
        ax = sns.heatmap(
            confusion_mat_display, 
            annot=True, 
            fmt=fmt, 
            cmap="viridis", 
            xticklabels=labels, 
            yticklabels=labels
        )
        
        plt.title(f"Confusion Matrix{title_suffix}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        # Improve readability for larger confusion matrices
        if len(confusion_mat) > 5:
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=45)
        
        plt.tight_layout()
        
        return plt.gcf()

    def _create_metrics_table_fig(self, metrics):
        """Create and return a figure with a metrics table."""
        # Extract main metrics
        main_metrics = {k: v for k, v in metrics.items() 
                      if k in ['accuracy', 'precision', 'recall', 'f1']}
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Hide axes
        ax.axis('off')
        
        # Create a table with metrics
        cell_text = [[f"{v:.4f}"] for v in main_metrics.values()]
        
        # Add colors to the table cells based on performance
        cell_colors = []
        for val in main_metrics.values():
            # Color gradient based on value
            if val >= 0.90:
                color = '#c6ecc6'  # Light green for excellent
            elif val >= 0.80:
                color = '#d6ecc6'  # Yellowish-green for good
            elif val >= 0.70:
                color = '#eeeeaa'  # Light yellow for fair
            else:
                color = '#f5c6c6'  # Light red for poor
            cell_colors.append([color])
        
        table = ax.table(
            cellText=cell_text, 
            rowLabels=list(main_metrics.keys()),
            colLabels=["Value"],
            loc='center', 
            cellLoc='center',
            cellColours=cell_colors
        )
        
        # Adjust table properties for better readability
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title("Overall Performance Metrics", fontsize=12, pad=20)
        plt.tight_layout()
        
        return fig
        
    def _create_per_class_metrics_fig(self, metrics):
        """Create and return a figure with per-class metrics."""
        # Extract per-class metrics
        per_class = metrics.get('per_class', {})
        
        if not per_class:
            return None
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(per_class).T
        
        # Select only the numeric columns for plotting
        metric_cols = ['precision', 'recall', 'f1-score']
        plot_df = df[metric_cols]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot heatmap with improved aesthetics
        ax = sns.heatmap(
            plot_df, 
            annot=True, 
            cmap="YlGnBu", 
            fmt='.3f', 
            vmin=0, 
            vmax=1, 
            linewidths=.5
        )
        
        # Adjust color bar
        cbar = ax.collections[0].colorbar
        cbar.set_label('Score Value')
        
        plt.title("Per-Class Performance Metrics", fontsize=14)
        plt.tight_layout()
        
        return plt.gcf()
        
    def _create_summary_text_fig(self, timestamp, metrics):
        """Create a figure with summary text."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        
        text = (
            f"Model Evaluation Report\n"
            f"=====================\n\n"
            f"Model: {self.model_name}\n"
            f"Generated: {timestamp}\n\n"
            f"Summary:\n"
            f"- Accuracy: {metrics['accuracy']:.4f}\n"
            f"- F1 Score (Macro): {metrics['f1']:.4f}\n"
            f"- Precision (Macro): {metrics['precision']:.4f}\n"
            f"- Recall (Macro): {metrics['recall']:.4f}\n"
        )
        
        # Add ROC AUC if available
        if 'roc_auc' in metrics:
            text += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
        
        # Add class information
        if self.class_names:
            text += f"\nClasses: {', '.join(self.class_names)}"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, 
               fontsize=12, va='center', family='monospace')
               
        return fig

    def _create_class_distribution_fig(self, all_labels):
        """Create a figure showing the distribution of classes in the dataset."""
        # Count occurrences of each class
        unique_labels = sorted(list(set(all_labels)))
        counts = [all_labels.count(label) for label in unique_labels]
        
        # Get class names
        class_names = self.class_names if self.class_names else [f"Class {i}" for i in unique_labels]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot bar chart with improved aesthetics
        bars = plt.bar(
            class_names, 
            counts, 
            color=self.color_palette,
            edgecolor='black', 
            alpha=0.8
        )
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
        
        # Add a horizontal line for the average count
        avg_count = sum(counts) / len(counts)
        plt.axhline(y=avg_count, color='r', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_count:.1f}')
        
        plt.title("Class Distribution in Test Data", fontsize=14)
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
        
    def _create_roc_curve_fig(self, metrics):
        """Create a figure with ROC curve if available."""
        if 'roc_curve' not in metrics:
            return None
            
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        roc_auc = metrics['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
        
    def _create_pr_curve_fig(self, metrics):
        """Create a figure with precision-recall curve if available."""
        if 'pr_curve' not in metrics:
            return None
            
        precision = metrics['pr_curve']['precision']
        recall = metrics['pr_curve']['recall']
        avg_precision = metrics['average_precision']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, 
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

    def generate_pdf_report(self, all_labels, all_preds, all_probs=None, output_path="model_evaluation_report.pdf"):
        """
        Generate a comprehensive PDF report with all evaluation metrics and visualizations.
        
        Args:
            all_labels: List of true labels
            all_preds: List of predicted labels
            all_probs: Array of prediction probabilities (optional)
            output_path: Path to save the PDF report
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', 
                   exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[INFO] Generating evaluation report at {output_path}...")
        
        # Create PDF
        with PdfPages(output_path) as pdf:
            # Add summary page
            summary_fig = self._create_summary_text_fig(timestamp, metrics)
            pdf.savefig(summary_fig)
            plt.close(summary_fig)
            
            # Add confusion matrix
            cm_fig = self._create_confusion_matrix_fig(metrics["confusion_matrix"])
            pdf.savefig(cm_fig)
            plt.close(cm_fig)
            
            # Add normalized confusion matrix for better interpretation
            if len(set(all_labels)) > 2:  # Only for multiclass
                norm_cm_fig = self._create_confusion_matrix_fig(metrics["confusion_matrix"], normalize=True)
                pdf.savefig(norm_cm_fig)
                plt.close(norm_cm_fig)
            
            # Add metrics table
            metrics_fig = self._create_metrics_table_fig(metrics)
            pdf.savefig(metrics_fig)
            plt.close(metrics_fig)
            
            # Add per-class metrics
            per_class_fig = self._create_per_class_metrics_fig(metrics)
            if per_class_fig:
                pdf.savefig(per_class_fig)
                plt.close(per_class_fig)
                
            # Add class distribution
            dist_fig = self._create_class_distribution_fig(all_labels)
            pdf.savefig(dist_fig)
            plt.close(dist_fig)
            
            # Add ROC curve if available (binary classification)
            roc_fig = self._create_roc_curve_fig(metrics)
            if roc_fig:
                pdf.savefig(roc_fig)
                plt.close(roc_fig)
                
            # Add PR curve if available (binary classification)
            pr_fig = self._create_pr_curve_fig(metrics)
            if pr_fig:
                pdf.savefig(pr_fig)
                plt.close(pr_fig)
            
            # Set PDF metadata
            pdf_info = pdf.infodict()
            pdf_info['Title'] = f'Model Evaluation Report: {self.model_name}'
            pdf_info['Author'] = 'ModelEvaluator'
            pdf_info['Subject'] = 'Machine Learning Model Evaluation'
            pdf_info['Keywords'] = 'machine learning, evaluation, metrics'
            pdf_info['CreationDate'] = datetime.now()
            pdf_info['ModDate'] = datetime.now()
            
        print(f"[INFO] Evaluation report saved as '{output_path}'")
        logger.info(f"Evaluation report saved as '{output_path}'")
        return output_path

    def evaluate_and_report(self, dataloader, output_path="model_evaluation_report.pdf"):
        """
        Convenience method to evaluate model and generate report in one step.
        
        Args:
            dataloader: DataLoader for test data
            output_path: Path to save the PDF report
            
        Returns:
            Dict: Evaluation metrics
        """
        # Run evaluation
        all_labels, all_preds, all_probs = self.evaluate(dataloader)
        
        # Generate report
        self.generate_pdf_report(all_labels, all_preds, all_probs, output_path)
        
        # Calculate and return metrics 
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        return metrics
        
    def display_interactive_report(self, dataloader):
        """
        Generate and display an interactive evaluation report in Jupyter notebook.
        
        Args:
            dataloader: DataLoader for test data
            
        Returns:
            Dict: Evaluation metrics
        """
        # Run evaluation
        all_labels, all_preds, all_probs = self.evaluate(dataloader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        # Display header
        display(HTML("""
        <style>
        .eval-header {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        .eval-section {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
            padding: 5px 0;
        }
        .metric-highlight {
            background-color: #e8f4f8;
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
            margin: 3px;
        }
        </style>
        """))
        
        display(HTML(f"<div class='eval-header'>{self.model_name} - Evaluation Report</div>"))
        
        # Overall metrics section
        display(HTML("<div class='eval-section'>Overall Performance Metrics</div>"))
        overall_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}"
            ]
        })
        display(overall_metrics.style.background_gradient(cmap='Blues', subset=['Value']).format({'Value': '{}'}).set_properties(**{'text-align': 'center'}))
        
        # Confusion matrix
        display(HTML("<div class='eval-section'>Confusion Matrix</div>"))
        plt.figure(figsize=(10, 8))
        cm_fig = self._create_confusion_matrix_fig(metrics["confusion_matrix"])
        plt.tight_layout()
        plt.show()
        
        # Per-class metrics section
        display(HTML("<div class='eval-section'>Per-Class Performance</div>"))
        per_class_metrics = pd.DataFrame(metrics['per_class']).T
        display(per_class_metrics.style.background_gradient(cmap='YlGnBu', subset=['precision', 'recall', 'f1-score']).format({'precision': '{:.4f}', 'recall': '{:.4f}', 'f1-score': '{:.4f}', 'support': '{:.0f}'}))
        
        # Class distribution
        display(HTML("<div class='eval-section'>Class Distribution</div>"))
        plt.figure(figsize=(10, 6))
        dist_fig = self._create_class_distribution_fig(all_labels)
        plt.tight_layout()
        plt.show()
        
        # ROC curve for binary classification
        if 'roc_auc' in metrics:
            display(HTML("<div class='eval-section'>ROC Curve</div>"))
            roc_fig = self._create_roc_curve_fig(metrics)
            plt.show()
            
            display(HTML("<div class='eval-section'>Precision-Recall Curve</div>"))
            pr_fig = self._create_pr_curve_fig(metrics)
            plt.show()
        
        return metrics