from contextlib import contextmanager
from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
from lxml.html.soupparser import fromstring as lxmlsoup
from lxml.html import HtmlElement
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    element_to_be_clickable,
    invisibility_of_element_located,
    invisibility_of_element,
)
from selenium.webdriver.common.by import By
from selenium.webdriver import (
    Firefox,
    FirefoxOptions,
    FirefoxProfile,
    Chrome,
    ChromeOptions,
)
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException as SeleniumTimeout,
)
from asyncio import sleep as sleep_async
from loguru import logger
from os.path import dirname as dirname
from time import time, sleep as sleep_sync
from typing import Any, Union, Self


class _SeleniumExtended:
    def __init__(self, **kwargs) -> None:
        if not any((isinstance(self, Firefox), isinstance(self, Chrome))):
            raise ImportError(
                "SeleniumExtended can only act as a superclass \
                for instances of webdriver.Firefox or webdriver.Chrome. \
                To inherit from _SeleniumExtended, make sure to also inherit \
                from one of those two classes."
            )
        self.waits = {}

    def get_xp_tree(self) -> etreeElement:
        return etreeHTML(
            text=self.page_source, parser=etreeHTMLParser(remove_comments=True)
        )

    def get_xp_tree_soup(self) -> HtmlElement:
        tree = lxmlsoup(self.page_source)
        assert tree is not None
        return tree

    def get_elem_xp(self, xpath: str, timeout: int = 10):
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(element_to_be_clickable((By.XPATH, xpath)))

    def wait_until_invisible_xp(self, xpath: str, timeout: int = 10):
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(
            invisibility_of_element_located((By.XPATH, xpath))
        )

    def wait_until_invisible(self, element: etreeElement, timeout: int = 10):
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(invisibility_of_element(element))

    async def get_elem_xp_tree(
        self, xpath: str, wait: int = 1, timeout: int | None = None, a_sync: bool = True
    ) -> list:
        t1 = time()
        while not (elem := (self.get_xp_tree()).xpath(xpath)):
            if timeout and (time() - t1 >= timeout):
                raise SeleniumTimeout
            if a_sync:
                await sleep_async(wait)
            else:
                sleep_sync(wait)
        return elem

    async def get_retry(
        self, url: str, sleep: float = 30, retries: int = 6, a_sync: bool = True
    ) -> None:
        for r in range(retries):
            try:
                self.get(url)
                return
            except WebDriverException:
                logger.warning(
                    f"Couldn't get {url}. Waiting {sleep} seconds for try #{r+1}/{retries}"
                )
                if a_sync:
                    await sleep_async(sleep)
                else:
                    sleep_sync(sleep)
        raise WebDriverException

    def cascade(self, *xpath: str, retries: int = 3):
        if len(xpath) == 1:
            return self.get_elem_xp(xpath[0])
        for i, xp in enumerate(xpath):
            try:
                self.get_elem_xp(xp).click()
            except SeleniumTimeout as e:
                if i == 0:
                    raise e
                for j in range(retries):
                    try:
                        self.get_elem_xp(xpath[i - 1]).click()
                        self.get_elem_xp(xpath[i]).click()
                    except SeleniumTimeout as f:
                        if j == retries - 1:
                            raise f
    
    @contextmanager
    def maintain_window(self, window: str | None = None):
        current_window = self.current_window_handle
        if window:
            self.switch_to.window(window)
        try:
            yield current_window
        finally:
            self.switch_to.window(current_window)


class SeleniumExtendedFirefox(Firefox, _SeleniumExtended):

    def __init__(
        self,
        headless: bool = False,
        user_agent: str | None = None,
        firefox_kwargs: dict[str, Any] | None = None,
    ) -> None:
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


class SeleniumExtendedChrome(Chrome, _SeleniumExtended):

    def __init__(
        self,
        headless: bool = False,
        user_agent: str | None = None,
        chrome_kwargs: dict | None = None,
    ) -> None:
        if chrome_kwargs is None:
            chrome_kwargs = {}
        if "options" in chrome_kwargs:
            options = chrome_kwargs["options"]
        else:
            options = ChromeOptions()
        if headless:
            if hl_index := next(
                i for i, v in enumerate(options.arguments) if v.startswith("--headless")
            ):
                options.arguments.pop(hl_index)
            options.add_argument("--headless=new")
        if user_agent is not None:
            if ua_index := next(
                i
                for i, v in enumerate(options.arguments)
                if v.startswith("--user-agent")
            ):
                options.arguments.pop(ua_index)
            options.add_argument(f"--user-agent={user_agent}")
        chrome_kwargs["options"] = options
        Chrome.__init__(self, **chrome_kwargs)
        _SeleniumExtended.__init__(self)

ExtendedSeleniumDriver = Union[SeleniumExtendedChrome, SeleniumExtendedFirefox]
