from cartoframes.auth import Credentials, set_default_credentials
from configparser import ConfigParser
from constants.global_constants import PROJECT_DIR


def set_creds(type="on-prem"):
    """
    Sets default Carto credentials

    Args:
        config (config, optional): Config file with Carto account information.
                                   Defaults to config.
        type (str, optional): cloud or on-prem. Defaults to "on-prem".
    """

    # Read in and establish credentials
    config = ConfigParser()
    config.read(PROJECT_DIR + "credentials.ini")

    if type == "cloud":
        # Main user credentials
        set_default_credentials(
            username=config["cloud"]["user"],
            api_key=config["cloud"]["api_key"],
        )

    elif type == "on-prem":
        # Main user credentials
        set_default_credentials(
            base_url="https://carto.tools.bain.com/user/{}".format(
                config["on-prem"]["user"]
            ),
            username=config["on-prem"]["user"],
            api_key=config["on-prem"]["api_key"],
        )

    else:
        raise ValueError("Type must be set to 'on-prem' or 'cloud'")


def get_creds(type="on-prem"):
    """
    Obtains a Carto credentials object to utilize for
    Carto operations (enrichment, isoline creation, i/o, etc.)

    Args:
        config (config, optional): Config file with Carto account information.
                                   Defaults to config.
        type (str, optional): cloud or on-prem. Defaults to "on-prem".

    Returns:
        creds: Carto credentials object to be utilized during
    """

    # Read in and establish credentials
    config = ConfigParser()
    config.read(PROJECT_DIR + "credentials.ini")

    if type == "cloud":
        # Main user credentials
        creds = Credentials(
            username=config["cloud"]["user"],
            api_key=config["cloud"]["api_key"],
        )

    elif type == "on-prem":
        # Main user credentials
        creds = Credentials(
            base_url="https://carto.tools.bain.com/user/{}".format(
                config["on-prem"]["user"]
            ),
            username=config["on-prem"]["user"],
            api_key=config["on-prem"]["api_key"],
        )

    else:
        raise ValueError("Type must be set to 'on-prem' or 'cloud'")

    return creds


def check_quota(required: int, available: int):
    """Print statement to articulate the quota results
    Paramters:
        required (int): Integer Carto credits required, output of dry_run=True
        available (int): Integer Carto credits remaining, output of check_quota

    Returns:
        None
    """
    assert isinstance(required, int)
    assert isinstance(available, int)

    if available >= required:
        print("Continue with geocoding")
        print(
            "{:,} credits are required, this is {:.1f}% of your remaining quota".format(
                required, 100 * required / float(available)
            )
        )
    else:
        print("Contact CARTO admisinstrator before continuing")
        print(
            "{} credits are required, you have only {} remaining".format(
                required, available
            )
        )
