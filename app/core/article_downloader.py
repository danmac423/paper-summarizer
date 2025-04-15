import requests
import re


class InvalidURLError(ValueError):
    pass


class DownloadError(Exception):
    pass


class NotPDFError(Exception):
    pass


class HTTPError(DownloadError):
    def __init__(self, status_code, message=""):
        self.status_code = status_code
        super().__init__(f"HTTP error {status_code}: {message}")


def is_url_valid(url: str) -> bool:
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


def download_pdf(url: str):
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
