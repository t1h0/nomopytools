# third-party imports
from selenium.webdriver import (
    Chrome,
    ChromeOptions,
)

# built-in imports
from os.path import dirname as dirname

# local imports
from .base import _SeleniumExtended


class ExtendedChrome(Chrome, _SeleniumExtended):

    def __init__(
        self,
        headless: bool = False,
        user_agent: str | None = None,
        chrome_kwargs: dict | None = None,
        humanoid: bool = False,
    ) -> None:
        """Extended Chrome driver.

        Args:
            headless (bool, optional): Whether to run headless. Defaults to False.
            user_agent (str | None, optional): The user agent to use. Defaults to None.
            chrome_kwargs (dict[str, Any] | None, optional): Additional chromedriver
                keyword arguments. Defaults to None.
            humanoid (bool, optional): Whether to show a humanoid browser signature.
                Defaults to False.
        """
        if chrome_kwargs is None:
            chrome_kwargs = {}
        if "options" in chrome_kwargs:
            options = chrome_kwargs["options"]
        else:
            options = ChromeOptions()

        args_to_add = {}

        if headless:
            args_to_add["headless"] = "new"

        if user_agent is not None:
            args_to_add["user-agent"] = user_agent

        if humanoid:
            # adding argument to disable the AutomationControlled flag
            args_to_add["disable-blink-features"] = "AutomationControlled"

            # exclude the collection of enable-automation switches
            options.add_experimental_option("excludeSwitches", ["enable-automation"])

            # turn-off userAutomationExtension
            options.add_experimental_option("useAutomationExtension", False)

        chrome_kwargs["options"] = self.add_chrome_kwargs(options, args_to_add)

        Chrome.__init__(self, **chrome_kwargs)
        _SeleniumExtended.__init__(self)

        if humanoid:
            # changing the property of the navigator value for webdriver to undefined
            self.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

    @classmethod
    def add_chrome_kwargs(
        cls, options: ChromeOptions, kwargs: dict[str, str]
    ) -> ChromeOptions:
        """Add or update ChromeOptions arguments.

        Args:
            options (ChromeOptions): The options to update.
            kwargs (dict[str, str]): The arguments to add/update.

        Returns:
            ChromeOptions: The updated ChromeOptions.
        """
        for key, value in kwargs.items():
            if old_index := next(
                (
                    i
                    for i, v in enumerate(options.arguments)
                    if v.startswith(f"--{key}")
                ),
                None,
            ):
                options.arguments.pop(old_index)
            options.add_argument(f"--{key}={value}")
        return options
