import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
import os

# ---- Configuration (edit these variables) ----
# SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"  # Change to your SMTP server
SMTP_PORT = 587  # Change to your SMTP port
SMTP_USERNAME = ""  # Your email address
SMTP_PASSWORD = ""  # Your email password or app password
SENDER_EMAIL = ""  # Sender email address

# Email Recipients
RECIPIENT_EMAILS = [
    # Add email addresses here
    "simomfb2@@gmail.com",
    "mohcinesahtani@gmail.com"
]

# Input file path
SUMMARY_JSON_PATH = "./detections_output/objects_classification_output/global_summary.json"

# Email subject template
EMAIL_SUBJECT = "Objects Counting Analysis Summary - {timestamp}"

def load_summary_data(json_path):
    """Load data from the global summary JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded summary data from {json_path}")
        return data
    except FileNotFoundError:
        print(f"Error: Summary file not found at {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}")
        return None
    except Exception as e:
        print(f"Error loading summary data: {e}")
        return None

def create_email_content(summary_data):
    """Create formatted email content from summary data"""
    
    # Extract key information
    analysis_summary = summary_data.get('analysis_summary', {})
    class_distribution = summary_data.get('class_distribution', {})
    video_summaries = summary_data.get('video_summaries', {})
    detailed_results = summary_data.get('detailed_results', [])
    
    # Create HTML email content
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .section-title {{ background-color: #e0e0e0; padding: 10px; border-radius: 3px; font-weight: bold; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
            .stat-box {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            .class-item {{ background-color: #f8f8f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .video-item {{ background-color: #f0f8ff; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #ddd; }}
            .folder-item {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #28a745; }}
            .zero-count {{ color: #dc3545; }}
            .positive-count {{ color: #28a745; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 Object Detection Analysis Summary</h1>
            <p><strong>Analysis Date:</strong> {analysis_summary.get('analysis_timestamp', 'Unknown')}</p>
            <p><strong>Analysis Method:</strong> {analysis_summary.get('analysis_method', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Overall Statistics</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{analysis_summary.get('total_hand_folders_processed', 0)}</div>
                    <div class="stat-label">Total Hand Folders</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{analysis_summary.get('folders_with_objects', 0)}</div>
                    <div class="stat-label">Folders with Objects</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{analysis_summary.get('total_objects_counted', 0)}</div>
                    <div class="stat-label">Total Objects Counted</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{analysis_summary.get('average_objects_per_folder', 0):.2f}</div>
                    <div class="stat-label">Avg Objects per Folder</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🏷️ Class Distribution</div>
            <div class="stats-grid">
    """
    
    # Add class distribution
    for class_name, count in class_distribution.items():
        html_content += f"""
                <div class="class-item">
                    <strong>{class_name}:</strong> {count} objects
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📹 Video Summaries</div>
    """
    
    # Add video summaries
    for video_name, video_data in video_summaries.items():
        video_meta = video_data.get('video_metadata', {})
        html_content += f"""
            <div class="video-item">
                <h3>🎬 {video_name}</h3>
                <p><strong>Date:</strong> {video_meta.get('start_date', 'Unknown')}</p>
                <p><strong>Chunk:</strong> {video_meta.get('chunk_number', 'Unknown')}</p>
                <p><strong>Total Objects:</strong> {video_data.get('total_objects_count', 0)}</p>
                <p><strong>Hand Folders:</strong> {video_data.get('total_hand_folders', 0)}</p>
                <p><strong>Folders with Objects:</strong> {video_data.get('hand_folders_with_objects', 0)}</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <div class="section-title">📁 Detailed Folder Results</div>
    """
    
    # Add detailed results
    for result in detailed_results:
        object_count = result.get('object_count', 0)
        count_class = "zero-count" if object_count == 0 else "positive-count"
        
        html_content += f"""
            <div class="folder-item">
                <h4>📂 {result.get('folder_name', 'Unknown')}</h4>
                <p><strong>Objects:</strong> <span class="{count_class}">{object_count}</span></p>
                <p><strong>Images:</strong> {result.get('images_total', 0)} total, {result.get('images_with_objects', 0)} with objects</p>
                <p><strong>Avg Objects per Frame:</strong> {result.get('average_objects_per_frame', 0):.2f}</p>
                <p><strong>Intersection Rate:</strong> {result.get('intersection_rate', 0):.1f}%</p>
                <p><strong>Avg Confidence:</strong> {result.get('average_confidence', 0):.3f}</p>
        """
        
        # Add class distribution for this folder
        class_dist = result.get('class_distribution', {})
        if class_dist:
            html_content += "<p><strong>Classes:</strong> "
            class_items = [f"{cls}: {count}" for cls, count in class_dist.items()]
            html_content += ", ".join(class_items) + "</p>"
        
        html_content += "</div>"
    
    html_content += """
        </div>
        
        <div class="footer" style="margin-top: 30px; padding: 15px; background-color: #f0f0f0; border-radius: 5px; text-align: center; color: #666;">
            <p>This report was automatically generated by the Object Detection Analysis System</p>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    </body>
    </html>
    """
    
    # Create plain text version
    text_content = f"""
Object Detection Analysis Summary
================================

Analysis Date: {analysis_summary.get('analysis_timestamp', 'Unknown')}
Analysis Method: {analysis_summary.get('analysis_method', 'Unknown')}

OVERALL STATISTICS:
- Total Hand Folders: {analysis_summary.get('total_hand_folders_processed', 0)}
- Folders with Objects: {analysis_summary.get('folders_with_objects', 0)}
- Total Objects Counted: {analysis_summary.get('total_objects_counted', 0)}
- Average Objects per Folder: {analysis_summary.get('average_objects_per_folder', 0):.2f}

CLASS DISTRIBUTION:
"""
    
    for class_name, count in class_distribution.items():
        text_content += f"- {class_name}: {count} objects\n"
    
    text_content += f"\nVIDEO SUMMARIES:\n"
    for video_name, video_data in video_summaries.items():
        video_meta = video_data.get('video_metadata', {})
        text_content += f"- {video_name}: {video_data.get('total_objects_count', 0)} objects, {video_data.get('hand_folders_with_objects', 0)}/{video_data.get('total_hand_folders', 0)} folders with objects\n"
    
    text_content += f"\nDETAILED FOLDER RESULTS:\n"
    for result in detailed_results:
        text_content += f"- {result.get('folder_name', 'Unknown')}: {result.get('object_count', 0)} objects, {result.get('images_with_objects', 0)}/{result.get('images_total', 0)} images with objects\n"
    
    text_content += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return html_content, text_content

def send_email(recipient_email, subject, html_content, text_content):
    """Send email to a single recipient"""
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach both HTML and text versions
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email to {recipient_email}: {e}")
        return False

def main():
    """Main function to send summary emails"""
    print("📧 Starting email summary distribution...")
    
    # Check configuration
    if not SMTP_USERNAME or not SMTP_PASSWORD or not SENDER_EMAIL:
        print("❌ Error: Please configure SMTP settings in the script")
        print("   - SMTP_USERNAME: Your email address")
        print("   - SMTP_PASSWORD: Your email password or app password")
        print("   - SENDER_EMAIL: Sender email address")
        return
    
    if not RECIPIENT_EMAILS:
        print("❌ Error: Please add recipient email addresses to RECIPIENT_EMAILS list")
        return
    
    # Load summary data
    summary_data = load_summary_data(SUMMARY_JSON_PATH)
    if not summary_data:
        print("❌ Failed to load summary data. Exiting.")
        return
    
    # Create email content
    print("📝 Creating email content...")
    html_content, text_content = create_email_content(summary_data)
    
    # Create email subject with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    subject = EMAIL_SUBJECT.format(timestamp=timestamp)
    
    print(f"📤 Sending emails to {len(RECIPIENT_EMAILS)} recipients...")
    
    # Send emails to all recipients
    successful_sends = 0
    for recipient in RECIPIENT_EMAILS:
        if send_email(recipient, subject, html_content, text_content):
            successful_sends += 1
    
    print(f"\n📊 Email Summary:")
    print(f"   Total recipients: {len(RECIPIENT_EMAILS)}")
    print(f"   Successful sends: {successful_sends}")
    print(f"   Failed sends: {len(RECIPIENT_EMAILS) - successful_sends}")
    
    if successful_sends == len(RECIPIENT_EMAILS):
        print("🎉 All emails sent successfully!")
    else:
        print("⚠️  Some emails failed to send. Check the error messages above.")

if __name__ == "__main__":
    main()
