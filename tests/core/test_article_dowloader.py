from app.core.article_downloader import (
    is_url_valid,
    download_pdf,
    InvalidURLError,
    DownloadError,
    HTTPError,
    NotPDFError,
)
import pytest
import requests
from unittest import mock


test_urls = [
    ("google", False),
    ("google.com", False),
    ("http://google.com", True),
    ("http://google", False),
]


@pytest.mark.parametrize("url,expected", test_urls)
def test_is_url_valid(url, expected):
    assert is_url_valid(url) == expected


@mock.patch("requests.get")
def test_dowload_pdf_valid(mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.content = b"%PDF-1.5..."
    mock_get.return_value = mock_response

    url = "http://example.com/valid.pdf"
    response = download_pdf(url)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/pdf"
    assert response.content == b"%PDF-1.5..."
    mock_get.assert_called_once_with(url, stream=True)


@mock.patch("requests.get")
def test_download_pdf_invalid_url(mock_get):
    url = "invalid-url"
    with pytest.raises(InvalidURLError, match="Provided URL is invalid: invalid-url"):
        download_pdf(url)
    mock_get.assert_not_called()


@mock.patch("requests.get")
def test_download_pdf_connection_error(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError(
        "Connection error occurred"
    )
    url = "http://example.com/error"
    with pytest.raises(
        DownloadError,
        match=f"There was a problem downloading URL: {url}. Details: Connection error occurred",
    ):
        download_pdf(url)

    mock_get.assert_called_once_with(url, stream=True)


@mock.patch("requests.get")
def test_download_pdf_http_error_not_found(mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Not Found"
    )
    mock_get.return_value = mock_response
    url = "http://example.com/not-found.pdf"
    with pytest.raises(HTTPError) as context:
        download_pdf(url)
    assert context.value.status_code == 404
    assert str(context.value) == "HTTP error 404: Not Found"


@mock.patch("requests.get")
def test_download_pdf_server_error(mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Server Error"
    )
    mock_get.return_value = mock_response
    url = "http://example.com/server-error.pdf"
    with pytest.raises(HTTPError) as context:
        download_pdf(url)
    assert context.value.status_code == 500
    assert str(context.value) == "HTTP error 500: Server Error"


@mock.patch("requests.get")
def test_download_pdf_not_pdf_content_type(mock_get):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.content = b"<html><body>Not a PDF</body></html>"
    mock_get.return_value = mock_response
    url = "http://example.com/not-a-pdf"
    with pytest.raises(NotPDFError) as context:
        download_pdf(url)
    assert (
        str(context.value)
        == "Downloaded resource is not a PDF file. Content type: text/html"
    )
