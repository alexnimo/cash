import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from app.models.video import Video
import logging
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str):
        """Initialize the report generator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.styles.add(ParagraphStyle(
            name='FrameTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=20
        ))

    def get_frame_metadata(self, frame_path: str) -> dict:
        """Extract metadata from a frame."""
        try:
            frame_path = Path(frame_path)
            frame_name = frame_path.name
            abs_path = frame_path.resolve()
            
            with PILImage.open(frame_path) as img:
                width, height = img.size
                # Extract timestamp from frame filename (e.g., frame_0010.jpg -> 10 seconds)
                timestamp = int(frame_name.split('_')[1].split('.')[0])
                return {
                    "Frame Name": frame_name,
                    "Timestamp": f"{timestamp} seconds",
                    "Resolution": f"{width}x{height}",
                    "File Path": str(abs_path),
                    "File Size": f"{os.path.getsize(frame_path) / 1024:.2f} KB"
                }
        except Exception as e:
            logger.error(f"Failed to extract frame metadata: {str(e)}")
            return {}

    def generate_frames_report(self, video: Video) -> str:
        """Generate a PDF report containing frames and their metadata."""
        try:
            logger.info(f"Starting frames report generation for video {video.id}")
            
            # Create output path
            output_path = self.output_dir / f"{video.id}_frames_report.pdf"
            logger.debug(f"Report will be saved to: {output_path}")
            
            # Check if video has frames
            if not hasattr(video, 'analysis') or not video.analysis or not hasattr(video.analysis, 'key_frames'):
                logger.error(f"Video {video.id} has no analysis or key_frames attribute")
                raise ValueError("No frames available for report generation")

            # Log frame information
            if not video.analysis.key_frames:
                logger.error("key_frames list is empty")
                raise ValueError("No frames available for report generation")
                
            logger.info(f"Found {len(video.analysis.key_frames)} frames to process")

            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Build the PDF content
            story = []
            
            # Add title
            title = Paragraph(f"Video Analysis Report - {video.id}", self.styles['Heading1'])
            story.append(title)
            story.append(Spacer(1, 30))
            
            # Add video information
            if video.metadata:
                video_info = [
                    f"Title: {video.metadata.title}",
                    f"Duration: {video.metadata.duration} seconds",
                    f"Resolution: {video.metadata.resolution[0]}x{video.metadata.resolution[1]}",
                    f"Format: {video.metadata.format}"
                ]
                for info in video_info:
                    story.append(Paragraph(info, self.styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Add frames section title
            frames_title = Paragraph("Extracted Frames", self.styles['Heading2'])
            story.append(frames_title)
            story.append(Spacer(1, 20))
            
            # Process each frame
            for frame_path in video.analysis.key_frames:
                logger.debug(f"Processing frame at path: {frame_path}")
                
                # Verify frame exists
                if not os.path.exists(frame_path):
                    logger.error(f"Frame file not found: {frame_path}")
                    continue

                # Get frame metadata
                metadata = self.get_frame_metadata(frame_path)
                if not metadata:
                    logger.error(f"Failed to get metadata for frame: {frame_path}")
                    continue

                # Add frame title
                frame_title = Paragraph(f"Frame: {metadata['Frame Name']}", self.styles['FrameTitle'])
                story.append(frame_title)
                
                # Add frame metadata
                metadata_text = []
                for key, value in metadata.items():
                    metadata_text.append(f"{key}: {value}")
                metadata_para = Paragraph("\n".join(metadata_text), self.styles['Metadata'])
                story.append(metadata_para)
                
                # Add frame image
                try:
                    img = Image(frame_path, width=6*inch, height=4*inch)
                    story.append(img)
                    logger.debug(f"Added image to report: {frame_path}")
                except Exception as e:
                    logger.error(f"Failed to add image to report: {str(e)}")
                    continue
                
                story.append(Spacer(1, 30))
            
            # Build the PDF
            logger.debug("Building final PDF document")
            doc.build(story)
            logger.info(f"Generated frames report at {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate frames report: {str(e)}", exc_info=True)
            raise
