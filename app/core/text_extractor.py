import os

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from app.core.exceptions import TextExtractionError
from app.core.singleton_meta import SingletonMeta


class MarkerTextExtractor(metaclass=SingletonMeta):
    """
    Singleton class to manage the Marker PDF converter.
    This class ensures that the converter is only initialized once,
    even if called from multiple threads.
    """

    def __init__(self):
        self.converter = self._initialize_converter()

    def _initialize_converter(self) -> PdfConverter:
        """
        Initializes the Marker PDF converter.
        Returns:
            PdfConverter: Initialized Marker PDF converter.
        Raises:
            TextExtractionError: If there is an error during converter initialization.
        """
        try:
            config = {
                "disable_image_extraction": True,
                "disable_links": True,
            }
            config_parser = ConfigParser(config)

            converter = converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config=config_parser.generate_config_dict(),
            )

            return converter
        except Exception as e:
            raise TextExtractionError(
                f"Failed to initialize Marker PDF converter: {e}"
            ) from e

    def extract_text_from_pdf_file(self, file_path: str) -> str:
        """
        Extracts text from a PDF file using the Marker PDF converter.
        Args:
            file_path (str): Path to the PDF file.
        Returns:
            str: Extracted text from the PDF file.
        Raises:
            TextExtractionError: If there is an error during text extraction.
        """
        if not os.path.exists(file_path):
            raise TextExtractionError(f"PDF file not found at: {file_path}")

        try:
            rendered = self.converter(file_path)
            text, _, _ = text_from_rendered(rendered)
            return text
        except Exception as e:
            raise TextExtractionError(
                f"Error during Marker PDF extraction for {file_path}: {e}"
            ) from e
