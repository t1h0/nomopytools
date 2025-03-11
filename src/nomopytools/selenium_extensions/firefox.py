# third-party imports
from selenium.webdriver import (
    Firefox,
    FirefoxOptions,
    FirefoxProfile,
)

# built-in imports
from os.path import dirname as dirname
from typing import Any

# local imports
from .base import _SeleniumExtended


class ExtendedFirefox(Firefox, _SeleniumExtended):

    def __init__(
        self,
        headless: bool = False,
        user_agent: str | None = None,
        firefox_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Extended Firefox driver.

        Args:
            headless (bool, optional): Whether to run headless. Defaults to False.
            user_agent (str | None, optional): The user agent to use. Defaults to None.
            firefox_kwargs (dict[str, Any] | None, optional): Additional geckodriver
                keyword arguments. Defaults to None.
        """
        if firefox_kwargs is None:
            firefox_kwargs = {}

        if "options" in firefox_kwargs:
            options = firefox_kwargs["options"]
        else:
            options = FirefoxOptions()

        if headless and "-headless" not in options.arguments:
            options.add_argument("-headless")

        if user_agent is not None:
            if options.profile is None:
                profile = FirefoxProfile()
                options.profile = profile
            options.profile.set_preference("general.useragent.override", user_agent)

        firefox_kwargs["options"] = options

        Firefox.__init__(self, **firefox_kwargs)
        _SeleniumExtended.__init__(self)
