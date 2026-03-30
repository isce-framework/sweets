import netrc
import os
from pathlib import Path

from rich.panel import Panel

from ._log import console
from ._types import Filename

NASA_HOST = "urs.earthdata.nasa.gov"
CDSE_HOST = "dataspace.copernicus.eu"


def setup_nasa_netrc(netrc_file: Filename = "~/.netrc"):
    """Prompt user for NASA username/password, store as attribute of ~/.netrc."""
    netrc_file = Path(netrc_file).expanduser()
    try:
        n = netrc.netrc(netrc_file)
        has_correct_permission = _file_is_0600(netrc_file)
        if not has_correct_permission:
            # User has a netrc file, but it's not set up correctly
            console.print(
                "Your ~/.netrc file does not have the correct"
                " permissions.\n[underline]Changing permissions to 0600"
                " (read/write for user only).",
                style="bold",
            )
            os.chmod(netrc_file, 0o600)
        # Check account exists, as well is having username and password
        _has_nasa_entry = (
            NASA_HOST in n.hosts
            and n.authenticators(NASA_HOST)[0]  # type: ignore
            and n.authenticators(NASA_HOST)[2]  # type: ignore
        )
        if _has_nasa_entry:
            return
    except FileNotFoundError:
        # User doesn't have a netrc file, make one
        console.print("No ~/.netrc file found, creating one.", style="bold")
        Path(netrc_file).write_text("")
        n = netrc.netrc(netrc_file)

    username, password = _get_username_pass()
    # Add NASA account to netrc file
    n.hosts[NASA_HOST] = (username, "", password)
    console.print(f"Saving credentials to {netrc_file} (machine={NASA_HOST}).")
    with open(netrc_file, "w") as f:
        f.write(str(n))
    # Set permissions to 0600 (read/write for user only)
    # https://www.ibm.com/docs/en/aix/7.1?topic=formats-netrc-file-format-tcpip
    os.chmod(netrc_file, 0o600)


def _file_is_0600(filename: Filename):
    """Check that a file has 0600 permissions (read/write for user only)."""
    return oct(Path(filename).stat().st_mode)[-4:] == "0600"


def _get_username_pass():
    """If netrc is not set up, get username/password via command line input."""
    console.print(
        Panel("Please enter NASA Earthdata credentials to download ASF-hosted data.")
    )
    console.print(
        "See the https://urs.earthdata.nasa.gov/users/new for signup info",
        style="link https://urs.earthdata.nasa.gov/users/new",
    )

    username = console.input("Username: ")
    password = console.input("Password (will not be displayed): ", password=True)
    return username, password


def setup_cdse_netrc(netrc_file: Filename = "~/.netrc"):
    """Prompt user for CDSE username/password if not in ~/.netrc or env vars."""
    netrc_file = Path(netrc_file).expanduser()

    # Check environment variables first
    env_user = os.getenv("CDSE_USERNAME")
    env_pass = os.getenv("CDSE_PASSWORD")
    if env_user and env_pass:
        # Ensure they are in .netrc too
        _ensure_netrc_entry(netrc_file, CDSE_HOST, env_user, env_pass)
        return

    try:
        n = netrc.netrc(netrc_file)
        has_correct_permission = _file_is_0600(netrc_file)
        if not has_correct_permission:
            os.chmod(netrc_file, 0o600)
        _has_cdse_entry = (
            CDSE_HOST in n.hosts
            and n.authenticators(CDSE_HOST)[0]  # type: ignore
            and n.authenticators(CDSE_HOST)[2]  # type: ignore
        )
        if _has_cdse_entry:
            return
    except FileNotFoundError:
        console.print("No ~/.netrc file found, creating one.", style="bold")
        Path(netrc_file).write_text("")
        os.chmod(netrc_file, 0o600)
        n = netrc.netrc(netrc_file)

    username, password = _get_cdse_username_pass()
    n.hosts[CDSE_HOST] = (username, "", password)
    console.print(f"Saving credentials to {netrc_file} (machine={CDSE_HOST}).")
    with open(netrc_file, "w") as f:
        f.write(str(n))
    os.chmod(netrc_file, 0o600)


def _ensure_netrc_entry(
    netrc_file: Path, host: str, username: str, password: str
):
    """Append a netrc entry for `host` if it doesn't already exist."""
    try:
        if netrc_file.exists():
            nrc = netrc.netrc(netrc_file)
            if nrc.authenticators(host):
                return
    except netrc.NetrcParseError:
        pass

    with open(netrc_file, "a") as f:
        f.write(f"\nmachine {host} login {username} password {password}\n")
    netrc_file.chmod(0o600)


def _get_cdse_username_pass():
    """If CDSE netrc is not set up, get username/password via command line."""
    console.print(
        Panel(
            "Please enter CDSE credentials to download data from the "
            "Copernicus Data Space Ecosystem."
        )
    )
    console.print(
        "See https://dataspace.copernicus.eu/ for signup info",
        style="link https://dataspace.copernicus.eu/",
    )

    username = console.input("CDSE Username (email): ")
    password = console.input("CDSE Password (will not be displayed): ", password=True)
    return username, password
