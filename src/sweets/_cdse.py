"""Download Sentinel-1 SLC granules from the Copernicus Data Space Ecosystem (CDSE).

This module provides an alternative to the ASF-based download in download.py.
It searches the CDSE OData catalog by granule name and downloads the product zip.

CDSE credentials (username/password) are required and can be provided via:
  - Environment variables: CDSE_USERNAME and CDSE_PASSWORD
  - The ~/.netrc file with machine: dataspace.copernicus.eu

References:
  - https://documentation.dataspace.copernicus.eu/APIs/OData.html
  - https://documentation.dataspace.copernicus.eu/APIs/Token.html
"""

from __future__ import annotations

import netrc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests

from ._log import get_log
from ._types import Filename

logger = get_log(__name__)

CDSE_HOST = "dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"

# Retry settings
MAX_RETRIES = 3
RETRY_START_WAIT = 10  # seconds
RETRY_INCREMENT = 10  # seconds


def get_cdse_credentials(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve CDSE credentials from arguments, environment, or ~/.netrc.

    Parameters
    ----------
    username : str, optional
        CDSE username. Falls back to CDSE_USERNAME env var, then ~/.netrc.
    password : str, optional
        CDSE password. Falls back to CDSE_PASSWORD env var, then ~/.netrc.

    Returns
    -------
    tuple[str, str]
        (username, password)
    """
    if username and password:
        return username, password

    env_user = os.getenv("CDSE_USERNAME")
    env_pass = os.getenv("CDSE_PASSWORD")
    if env_user and env_pass:
        return env_user, env_pass

    try:
        nrc = netrc.netrc()
        auth = nrc.authenticators(CDSE_HOST)
        if auth:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    raise ValueError(
        "CDSE credentials not found. Provide them via:\n"
        "  1. CDSE_USERNAME and CDSE_PASSWORD environment variables\n"
        "  2. ~/.netrc entry for machine dataspace.copernicus.eu"
    )


def ensure_cdse_credentials(
    username: Optional[str] = None,
    password: Optional[str] = None,
    host: str = CDSE_HOST,
) -> None:
    """Ensure CDSE credentials are available in ~/.netrc.

    CDSE username and password may be provided by, in order of preference:
       * ``CDSE_USERNAME`` and ``CDSE_PASSWORD`` environment variables
       * ``~/.netrc`` entry for machine dataspace.copernicus.eu

    If credentials are provided via env vars but ~/.netrc does not
    contain an entry for CDSE, the entry will be appended to ~/.netrc.
    """
    if username is None:
        username = os.getenv("CDSE_USERNAME")
    if password is None:
        password = os.getenv("CDSE_PASSWORD")

    netrc_file = Path.home() / ".netrc"

    cdse_in_netrc = False
    if netrc_file.exists():
        try:
            nrc = netrc.netrc(netrc_file)
            if nrc.authenticators(host):
                cdse_in_netrc = True
        except netrc.NetrcParseError:
            pass

    if username and password and not cdse_in_netrc:
        with open(netrc_file, "a") as f:
            f.write(f"\nmachine {host} login {username} password {password}\n")
        netrc_file.chmod(0o600)

    try:
        get_cdse_credentials(username, password)
    except ValueError:
        raise ValueError(
            f"Please provide valid CDSE credentials via {netrc_file}, "
            f"the CDSE_USERNAME and CDSE_PASSWORD environment variables, "
            f"or add an entry for machine {host} to your ~/.netrc file.\n"
            f"Register for a free account at https://dataspace.copernicus.eu/"
        )


def get_cdse_access_token(username: str, password: str) -> str:
    """Obtain an access token from the CDSE identity provider."""
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }
    response = requests.post(CDSE_TOKEN_URL, data=data, timeout=60)
    response.raise_for_status()
    return response.json()["access_token"]


def search_cdse_by_granule_name(granule_name: str) -> dict:
    """Search the CDSE OData catalog for a Sentinel-1 SLC by granule name.

    Parameters
    ----------
    granule_name : str
        Sentinel-1 granule / scene name (without .SAFE or .zip extension).

    Returns
    -------
    dict
        Product entry from the CDSE OData response containing ``Id`` and ``Name``.

    Raises
    ------
    LookupError
        If the product is not found on CDSE.
    """
    granule_name = granule_name.replace(".zip", "").replace(".SAFE", "")
    safe_name = f"{granule_name}.SAFE"
    query = f"{CDSE_ODATA_URL}?$filter=Name eq '{safe_name}'"

    response = requests.get(query, timeout=120)
    response.raise_for_status()
    results = response.json().get("value", [])

    if not results:
        raise LookupError(
            f"Product '{safe_name}' not found in CDSE catalog. Query: {query}"
        )
    return results[0]


def download_single_slc_from_cdse(
    granule_name: str,
    access_token: str,
    output_dir: Filename = ".",
    max_retries: int = MAX_RETRIES,
) -> str:
    """Download a single Sentinel-1 SLC product from CDSE.

    Tries the compressed endpoint (/$zip) first, then falls back to
    the natively compressed archive (/$value).

    Parameters
    ----------
    granule_name : str
        Sentinel-1 granule / scene name.
    access_token : str
        CDSE bearer access token.
    output_dir : Filename
        Directory to save the downloaded zip file.
    max_retries : int
        Number of download attempts before raising an error.

    Returns
    -------
    str
        The filename of the downloaded zip file.
    """
    granule_name = granule_name.replace(".zip", "").replace(".SAFE", "")

    product = search_cdse_by_granule_name(granule_name)
    product_id = product["Id"]

    # Try compressed endpoint first (/$zip), fall back to /$value
    download_url_zip = f"{CDSE_DOWNLOAD_URL}({product_id})/$zip"
    download_url_value = f"{CDSE_DOWNLOAD_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}

    out_filename = f"{granule_name}.zip"
    out_path = Path(output_dir) / out_filename

    def _do_download(url: str) -> str:
        """Perform the actual download from a given URL."""
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 16):
                if chunk:
                    f.write(chunk)

        if out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)
            raise requests.RequestException("Downloaded file is empty")

        logger.info(
            f"Downloaded {out_filename} from CDSE"
            f" ({out_path.stat().st_size / 1e6:.1f} MB)"
        )
        return out_filename

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"CDSE download attempt #{attempt} for {granule_name}")
        try:
            try:
                return _do_download(download_url_zip)
            except requests.HTTPError as e:
                out_path.unlink(missing_ok=True)
                if e.response is not None and e.response.status_code == 404:
                    logger.info(
                        "Compressed format not available, falling back to"
                        " uncompressed..."
                    )
                    try:
                        return _do_download(download_url_value)
                    except requests.RequestException:
                        out_path.unlink(missing_ok=True)
                        raise
                raise
            except requests.RequestException:
                out_path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            last_exc = exc
            wait_time = RETRY_START_WAIT + RETRY_INCREMENT * (attempt - 1)
            if attempt < max_retries:
                logger.warning(
                    f"Attempt #{attempt} failed: {exc}. "
                    f"Waiting {wait_time}s before retry..."
                )
                import time

                time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to download {granule_name} from CDSE after {max_retries} attempts"
    ) from last_exc


def download_slcs_from_cdse(
    slc_ids: list[str],
    username: Optional[str] = None,
    password: Optional[str] = None,
    output_dir: Filename = ".",
    max_workers: int = 3,
) -> list[str]:
    """Download multiple Sentinel-1 SLC granules from CDSE.

    Parameters
    ----------
    slc_ids : list[str]
        List of Sentinel-1 granule / scene names (ASF-style).
    username : str, optional
        CDSE username. Resolved from env/netrc if not provided.
    password : str, optional
        CDSE password. Resolved from env/netrc if not provided.
    output_dir : Filename
        Directory to save downloaded zip files.
    max_workers : int
        Number of parallel download threads.

    Returns
    -------
    list[str]
        List of downloaded zip filenames.
    """
    cdse_user, cdse_pass = get_cdse_credentials(username, password)
    access_token = get_cdse_access_token(cdse_user, cdse_pass)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _download_one(slc_id: str) -> str:
        return download_single_slc_from_cdse(
            slc_id,
            access_token=access_token,
            output_dir=str(output_dir),
        )

    n = len(slc_ids)
    logger.info(f"Downloading {n} SLCs from CDSE to {output_dir}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_download_one, slc_ids))

    return results
