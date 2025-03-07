# third-party imports
from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
from lxml.html import HtmlElement
from bs4 import BeautifulSoup
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
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.shadowroot import ShadowRoot
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException as SeleniumTimeout,
    NoSuchElementException,
)

# built-in imports
from contextlib import contextmanager
from asyncio import sleep as sleep_async
from loguru import logger
from os.path import dirname as dirname
from time import time, sleep as sleep_sync
from typing import Any, Literal, Union, Callable, overload
import warnings


class _SeleniumExtended:
    def __init__(self, **kwargs) -> None:
        if not any((isinstance(self, Firefox), isinstance(self, Chrome))):
            raise ImportError(
                "SeleniumExtended can only act as a superclass"
                " for instances of webdriver.Firefox or webdriver.Chrome."
                " To inherit from _SeleniumExtended, make sure to also inherit"
                " from one of those two classes."
            )
        self.waits = {}

    def get_xp_tree(self, soup: bool = False) -> etreeElement:
        return etreeHTML(
            text=(
                str(BeautifulSoup(self.page_source, "lxml"))
                if soup
                else self.page_source
            ),
            parser=etreeHTMLParser(remove_comments=True),
        )

    def get_xp_tree_soup(self) -> HtmlElement:
        warnings.warn(DeprecationWarning("Use get_xp_tree(soup=True) instead."))
        return self.get_xp_tree(soup=True)

    def get_elem_xp(self, xpath: str, timeout: int = 10) -> WebElement:
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(element_to_be_clickable((By.XPATH, xpath)))

    def wait_until_invisible_xp(self, xpath: str, timeout: int = 10) -> WebElement:
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(
            invisibility_of_element_located((By.XPATH, xpath))
        )

    def wait_until_invisible(
        self, element: etreeElement, timeout: int = 10
    ) -> WebElement:
        if timeout not in self.waits:
            self.waits[timeout] = WebDriverWait(self, timeout)
        return self.waits[timeout].until(invisibility_of_element(element))

    async def get_elem_xp_tree(
        self,
        xpath: str,
        timeout: int | None = None,
        wait: int | None = None,
        soup: bool = False,
    ) -> list[etreeElement]:
        return await self.timeout(
            func=lambda: self.get_xp_tree(soup=soup).xpath(xpath),
            timeout=timeout,
            wait=wait,
        )

    @overload
    async def get_elem_js(
        self,
        selector: str,
        timeout: int | None = None,
        wait: int | None = None,
        select_all: Literal[False] = False,
    ) -> WebElement: ...

    @overload
    async def get_elem_js(
        self,
        selector: str,
        timeout: int | None = None,
        wait: int | None = None,
        select_all: Literal[True] = ...,
    ) -> list[WebElement]: ...

    async def get_elem_js(
        self,
        selector: str,
        timeout: int | None = None,
        wait: int | None = None,
        select_all: bool = False,
    ) -> WebElement | list[WebElement]:
        return await self.timeout(
            func=lambda: self.execute_script(
                f"return document.querySelector{"All" if select_all else ""}('{selector}');"
            ),
            timeout=timeout,
            wait=wait,
        )

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[True] = True,
        timeout: int | None = None,
        wait: int | None = None,
    ) -> list[WebElement]: ...

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[False] = ...,
        timeout: int | None = None,
        wait: int | None = None,
    ) -> WebElement: ...

    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: bool = True,
        timeout: int | None = None,
        wait: int | None = None,
    ) -> list[WebElement] | WebElement:
        shadow_root = await self.get_elem_js(
            selector=shadow_root_selector, timeout=timeout, wait=wait
        ).shadow_root
        return await self.timeout(
            lambda: self.search_shadow_root(
                shadow_root=shadow_root,
                elem_css_selector=elem_css_selector,
                get_all=get_all,
            ),
            timeout=timeout,
            wait=wait,
        )

    @overload
    def search_shadow_root(
        self,
        shadow_root: ShadowRoot,
        elem_css_selector: str,
        get_all: Literal[True] = True,
    ) -> list[WebElement]: ...
    @overload
    def search_shadow_root(
        self,
        shadow_root: ShadowRoot,
        elem_css_selector: str,
        get_all: Literal[False] = ...,
    ) -> WebElement | None: ...

    def search_shadow_root(
        self, shadow_root: ShadowRoot, elem_css_selector: str, get_all: bool = True
    ) -> list[WebElement] | WebElement | None:
        try:
            return (shadow_root.find_elements if get_all else shadow_root.find_element)(
                By.CSS_SELECTOR, elem_css_selector
            )
        except NoSuchElementException:
            return [] if get_all else None

    async def timeout(
        self, func: Callable, timeout: int | None = None, wait: int | None = None
    ) -> Any:
        if timeout is None:
            timeout = 0

        t1 = time()
        while not (result := func()):
            if time() - t1 >= timeout:
                raise SeleniumTimeout
            self.sleep(1 if wait is None else wait)
        return result

    # async def get_elem_xp_tree_async(
    #     self, xpath: str, timeout: int | None = None, wait: int = 1, soup: bool = False
    # ) -> list:
    #     t1 = time()
    #     while not (elem := (self.get_xp_tree(soup=soup)).xpath(xpath)):
    #         if timeout and (time() - t1 >= timeout):
    #             raise SeleniumTimeout
    #         await self.sleep(wait)
    #     return elem

    def get_retry(self, url: str, wait: float = 30, retries: int = 6) -> None:
        for r in range(retries):
            try:
                self.get(url)
                return
            except WebDriverException:
                logger.warning(
                    f"Couldn't get {url}. Waiting {wait} seconds for try #{r+1}/{retries}"
                )
                sleep_sync(wait)
        raise WebDriverException

    async def get_retry_async(
        self, url: str, wait: float = 30, retries: int = 6
    ) -> None:
        for r in range(retries):
            try:
                self.get(url)
                return
            except WebDriverException:
                logger.warning(
                    f"Couldn't get {url}. Waiting {wait} seconds for try #{r+1}/{retries}"
                )
                await self.sleep(wait)
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

    async def sleep(self, seconds: float) -> None:
        with self.maintain_window():
            await sleep_async(seconds)

    @contextmanager
    def maintain_window(self, window: str | None = None):
        original_window = self.current_window_handle
        if window:
            self.switch_to.window(window)
        try:
            yield original_window
        finally:
            self.switch_to.window(original_window)

    @contextmanager
    def own_window(
        self, typ: Literal["tab", "window"] = "tab", close_after: bool = False
    ):
        original_window = self.current_window_handle
        self.switch_to.new_window(typ)
        new_window = self.current_window_handle
        try:
            yield new_window
        finally:
            if close_after:
                self.switch_to.window(new_window)
                self.close()
            self.switch_to.window(original_window)


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
                (
                    i
                    for i, v in enumerate(options.arguments)
                    if v.startswith("--user-agent")
                ),
                None,
            ):
                options.arguments.pop(ua_index)
            options.add_argument(f"--user-agent={user_agent}")

        chrome_kwargs["options"] = options

        Chrome.__init__(self, **chrome_kwargs)
        _SeleniumExtended.__init__(self)


ExtendedSeleniumDriver = Union[SeleniumExtendedChrome, SeleniumExtendedFirefox]
