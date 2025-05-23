import streamlit as st
import tempfile
import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("video_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Streamlit app for video upload and playback."""
    st.set_page_config(page_title="Video Upload App", layout="wide")
    st.title("Upload and Play Video")
    st.markdown("Upload an MP4, AVI, or MOV video file to view it.")

    # Video file uploader
    video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        try:
            # Create a temporary file to store the uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(video_file.read())
                temp_file_path = temp_file.name
                logger.info(f"Video uploaded and saved to temporary file: {temp_file_path}")

            # Verify the video file
            if os.path.exists(temp_file_path):
                # Display the video
                with open(temp_file_path, "rb") as file:
                    video_bytes = file.read()
                st.video(video_bytes, format=video_file.type)
                st.success("Video loaded and displayed successfully!")
                logger.info("Video displayed in Streamlit")
            else:
                st.error("Failed to save the uploaded video.")
                logger.error(f"Video file not found: {temp_file_path}")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Error processing video: {str(e)}")

        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Deleted temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")
    else:
        st.info("Please upload a video file.")

if __name__ == "__main__":
    main()