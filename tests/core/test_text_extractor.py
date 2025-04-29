import pytest

from app.core.singleton_meta import SingletonMeta
from app.core.text_extractor import MarkerTextExtractor, TextExtractionError


@pytest.fixture(autouse=True)
def reset_singleton_meta_instances():
    with SingletonMeta._lock:
        SingletonMeta._instances = {}

    yield


def test_marker_text_extractor_initializes_on_first_call(mocker):
    mock_create_model_dict = mocker.patch("app.core.text_extractor.create_model_dict")
    mock_pdf_converter_class = mocker.patch("app.core.text_extractor.PdfConverter")

    MarkerTextExtractor()

    mock_pdf_converter_class.assert_called_once_with(
        artifact_dict=mock_create_model_dict.return_value, config=mocker.ANY
    )


def test_marker_text_extractor_returns_same_instance_on_subsequent_calls(mocker):
    mock_create_model_dict = mocker.patch("app.core.text_extractor.create_model_dict")
    mock_pdf_converter_class = mocker.patch("app.core.text_extractor.PdfConverter")

    converter_1 = MarkerTextExtractor()
    converter_2 = MarkerTextExtractor()
    converter_3 = MarkerTextExtractor()

    mock_pdf_converter_class.assert_called_once_with(
        artifact_dict=mock_create_model_dict.return_value, config=mocker.ANY
    )

    assert converter_1 is converter_2 is converter_3


def test_marker_text_extractor_initialization_error(mocker):
    mock_create_model_dict = mocker.patch("app.core.text_extractor.create_model_dict")
    mock_pdf_converter_class = mocker.patch("app.core.text_extractor.PdfConverter")
    marker_initialization_error = RuntimeError("Simulated error during model loading")
    mock_pdf_converter_class.side_effect = marker_initialization_error

    with pytest.raises(TextExtractionError) as excinfo:
        MarkerTextExtractor()

    assert "Failed to initialize Marker PDF converter" in str(excinfo.value)
    assert excinfo.value.__cause__ is marker_initialization_error
    mock_pdf_converter_class.assert_called_once_with(
        artifact_dict=mock_create_model_dict.return_value, config=mocker.ANY
    )
    assert not SingletonMeta._instances


def test_extract_from_pdf_file_raises_error_if_file_does_not_exist(mocker):
    mocker.patch("app.core.text_extractor.create_model_dict")
    mocker.patch("app.core.text_extractor.PdfConverter")
    mocker.patch("app.core.text_extractor.os.path.exists", return_value=False)

    converter = MarkerTextExtractor()

    with pytest.raises(TextExtractionError) as excinfo:
        converter.extract_text_from_pdf_file("non_existent_file.pdf")

    assert "PDF file not found at: non_existent_file.pdf" in str(excinfo.value)


def test_marker_text_extractor_extract_succeeds_with_valid_file(mocker):
    mock_os_path_exists = mocker.patch(
        "app.core.text_extractor.os.path.exists", return_value=True
    )

    mocker.patch("app.core.text_extractor.create_model_dict")
    mock_pdf_converter_class = mocker.patch("app.core.text_extractor.PdfConverter")

    mock_converter_instance = mocker.Mock()
    mock_pdf_converter_class.return_value = mock_converter_instance

    mock_rendered_output = mocker.Mock()
    mock_converter_instance.return_value = mock_rendered_output

    mock_extracted_text = "Extracted text content from PDF."
    mock_text_from_rendered = mocker.patch(
        "app.core.text_extractor.text_from_rendered",
        return_value=(
            mock_extracted_text,
            None,
            None,
        ),
    )

    extractor_instance = MarkerTextExtractor()

    file_path_to_test = "valid_file.pdf"
    result = extractor_instance.extract_text_from_pdf_file(file_path_to_test)

    mock_os_path_exists.assert_called_once_with(file_path_to_test)
    mock_converter_instance.assert_called_once_with(file_path_to_test)
    mock_text_from_rendered.assert_called_once_with(mock_rendered_output)

    assert result == mock_extracted_text


def test_marker_text_extractor_extract_raises_error_on_conversion_failure(mocker):
    mocker.patch("app.core.text_extractor.os.path.exists", return_value=True)

    mocker.patch("app.core.text_extractor.create_model_dict")
    mock_pdf_converter_class = mocker.patch("app.core.text_extractor.PdfConverter")

    mock_converter_instance = mocker.Mock()
    mock_pdf_converter_class.return_value = mock_converter_instance

    conversion_error = RuntimeError("Simulated error during PDF conversion")
    mock_converter_instance.side_effect = conversion_error

    extractor_instance = MarkerTextExtractor()

    file_path_to_test = "valid_file.pdf"
    with pytest.raises(TextExtractionError) as excinfo:
        extractor_instance.extract_text_from_pdf_file(file_path_to_test)

    assert f"Error during Marker PDF extraction for {file_path_to_test}" in str(
        excinfo.value
    )
    assert excinfo.value.__cause__ is conversion_error
