import requests
import re


class InvalidURLError(ValueError):
    """Exception raised for invalid URLs."""

    pass


class DownloadError(Exception):
    """Exception raised for errors during the download process."""

    pass


class NotPDFError(Exception):
    """Exception raised for non-PDF downloads."""

    pass


class HTTPError(DownloadError):
    """Exception raised for HTTP errors during download."""

    def __init__(self, status_code, message=""):
        self.status_code = status_code
        super().__init__(f"HTTP error {status_code}: {message}")


def is_url_valid(url: str) -> bool:
    """
    Validate if the provided URL is well-formed.
    Args:
        url (str): The URL to validate.
    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    regex = re.compile(
        r"^(?:http)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return re.match(regex, url) is not None


def download_pdf(url: str) -> requests.Response:
    """
    Download a PDF file from the given URL.
    Args:
        url (str): The URL of the PDF file to download.
    Returns:
        requests.Response: The response object containing the PDF file.
    Raises:
        InvalidURLError: If the URL is invalid.
        DownloadError: If there is a problem with the download.
        HTTPError: If the response status code indicates an error.
        NotPDFError: If the downloaded file is not a PDF.
    """
    if not is_url_valid(url):
        raise InvalidURLError(f"Provided URL is invalid: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type")
        if content_type != "application/pdf":
            raise NotPDFError(
                f"Downloaded resource is not a PDF file. Content type: {content_type}"
            )
        return response
    except requests.exceptions.ConnectionError as e:
        raise DownloadError(f"There was a problem downloading URL: {url}. Details: {e}")
    except requests.exceptions.HTTPError as e:
        raise HTTPError(response.status_code, str(e))
    except NotPDFError:
        raise
    except Exception as e:
        raise DownloadError(
            f"Unexpected error while downloading URL: {url}. Details: {e}"
        )
