# third-party imports
from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
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
from typing import Any, Literal, Union, Callable, overload, TypeVar
import warnings

T = TypeVar("T")


class _SeleniumExtended:
    def __init__(self, **kwargs) -> None:
        """Extended Selenium Driver Superclass"""
        if not any((isinstance(self, Firefox), isinstance(self, Chrome))):
            raise ImportError(
                "SeleniumExtended can only act as a superclass"
                " for instances of webdriver.Firefox or webdriver.Chrome."
                " To inherit from _SeleniumExtended, make sure to also inherit"
                " from one of those two classes."
            )

    def get_xp_tree(self, soup: bool = False) -> etreeElement:
        """Get the current xpath tree.

        Args:
            soup (bool, optional): Whether to build the tree with BeautifulSoup.
                Defaults to False.

        Returns:
            etreeElement: The xpath tree.
        """
        return etreeHTML(
            text=(
                str(BeautifulSoup(self.page_source, "lxml"))
                if soup
                else self.page_source
            ),
            parser=etreeHTMLParser(remove_comments=True),
        )

    def get_xp_tree_soup(self) -> etreeElement:
        warnings.warn(DeprecationWarning("Use get_xp_tree(soup=True) instead."))
        return self.get_xp_tree(soup=True)

    def get_elem_xp(self, xpath: str, timeout: int = 10) -> WebElement:
        """Get an element using xpath.

        Args:
            xpath (str): The xpath to use.
            timeout (int, optional): Timeout for waiting. Defaults to 10.

        Returns:
            WebElement: The element.
        """
        return WebDriverWait(self, timeout).until(
            element_to_be_clickable((By.XPATH, xpath))
        )

    def wait_until_invisible_xp(
        self, xpath: str, timeout: int = 10
    ) -> WebElement | bool:
        """Wait until an element (located using xpath) is invisible.

        Args:
            xpath (str): The xpath to use for locating the element.
            timeout (int, optional): Timeout for waiting. Defaults to 10.

        Returns:
            WebElement | bool: The element or True if the element is invisible
                or not present or False.
        """
        return WebDriverWait(self, timeout).until(
            invisibility_of_element_located((By.XPATH, xpath))
        )

    def wait_until_invisible(
        self, element: etreeElement, timeout: int = 10
    ) -> WebElement | bool:
        """Wait until the element is invisible.

        Args:
            element (etreeElement): The element to wait for its invisibility.
            timeout (int, optional): Timeout for waiting. Defaults to 10.

        Returns:
            WebElement | bool: The element or True if the element is invisible
                or not present or False.
        """
        return WebDriverWait(self, timeout).until(invisibility_of_element(element))

    async def get_elem_xp_tree(
        self,
        xpath: str,
        timeout: int | None = None,
        soup: bool = False,
    ) -> list[etreeElement]:
        """Get an element using xpath on the current xpath tree.

        Args:
            xpath (str): The xpath to use.
            timeout (int | None, optional): Timeout for getting the element.
                If None, will not wait for it to appear but immediately check and return.
                Defaults to None.
            soup (bool, optional): Whether to use BeautifulSoup
                for constructing the xpath tree. Defaults to False.

        Returns:
            list[etreeElement]: List of elements matching the xpath.
        """
        return await self.timeout(
            func=lambda: self.get_xp_tree(soup=soup).xpath(xpath),
            timeout=timeout,
        )

    @overload
    async def get_elem_js(
        self,
        selector: str,
        get_all: Literal[True] = ...,
        timeout: int | None = None,
    ) -> list[WebElement]: ...

    @overload
    async def get_elem_js(
        self,
        selector: str,
        get_all: Literal[False] = False,
        timeout: int | None = None,
    ) -> WebElement | None: ...

    @overload
    async def get_elem_js(
        self,
        selector: str,
        get_all: Literal[False] = False,
        timeout: int = ...,
    ) -> WebElement: ...

    async def get_elem_js(
        self,
        selector: str,
        get_all: bool = True,
        timeout: int | None = None,
    ) -> list[WebElement] | WebElement | None:
        """Get element(s) using JavaScript.

        Args:
            selector (str): The selector to use.
            get_all (bool, optional): Whether to return all matches or just the first.
                Defaults to True.
            timeout (int | None, optional): Timeout for getting the element(s).
                If None, will not wait for it to appear but immediately check and return.
                Defaults to None.

        Returns:
            list[WebElement] | WebElement | None: The matched element(s) or None if
                none found.
        """
        return await self.timeout(
            func=lambda: self.execute_script(
                f"return document.querySelector{'All' if get_all else ''}('{selector}');"
            ),
            timeout=timeout,
        )

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[True] = True,
        timeout: int | None = None,
    ) -> list[WebElement]: ...

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[False] = ...,
        timeout: None = None,
    ) -> WebElement | None: ...

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[False] = ...,
        timeout: int = ...,
    ) -> WebElement: ...

    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: bool = True,
        timeout: int | None = None,
    ) -> list[WebElement] | WebElement | None:
        """Get an element inside a shadow root.

        Args:
            shadow_root_selector (str): The javascript selector for the shadow root.
            elem_css_selector (str): The CSS selector for the shadowed element.
            get_all (bool, optional): Whether to return all matches or just the first.
                Defaults to True.
            timeout (int | None, optional): Timeout for getting the element(s).
                If None, will not wait for it to appear but immediately check and return.
                Defaults to None.

        Returns:
            list[WebElement] | WebElement | None: The matched element(s) or None if
                None was found.
        """
        shadow_root = (
            await self.get_elem_js(
                selector=shadow_root_selector, timeout=timeout, get_all=False
            )
        ).shadow_root

        return await self.timeout(
            func=lambda: self.search_shadow_root(
                shadow_root=shadow_root,
                elem_css_selector=elem_css_selector,
                get_all=get_all,
            ),
            timeout=timeout,
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
        get_all: Literal[False],
    ) -> WebElement | None: ...

    def search_shadow_root(
        self, shadow_root: ShadowRoot, elem_css_selector: str, get_all: bool = True
    ) -> list[WebElement] | WebElement | None:
        """Searches a shadow root for one or all elements matching the selector.

        Args:
            shadow_root (ShadowRoot): The shadow root to search in.
            elem_css_selector (str): The CSS selector for the element(s).
            get_all (bool, optional): Whether to get all matches or just the first.
                Defaults to True.

        Returns:
            list[WebElement] | WebElement | None: The matched element(s) or
        """
        try:
            return (shadow_root.find_elements if get_all else shadow_root.find_element)(
                By.CSS_SELECTOR, elem_css_selector
            )
        except NoSuchElementException:
            return [] if get_all else None

    async def timeout(self, func: Callable[[], T], timeout: int | None = None) -> T:
        """Wait for func to return a truthy value until timeout.

        Args:
            func (Callable[[], T]): The function to wait for.
            timeout (int | None, optional): The timeout for waiting. If None, will
                return immediately. Defaults to None.

        Raises:
            SeleniumTimeout: If timeout is not None and time has run out.

        Returns:
            T: The return value of func.
        """
        t1 = time()
        while not (result := func()):
            if time() - t1 >= (timeout or 0):
                if timeout is None:
                    break
                raise SeleniumTimeout
            await self.sleep(1)
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

    async def get_retry(self, url: str, wait: float = 30, retries: int = 6) -> None:
        """Access a url with retry.

        Args:
            url (str): The url to access.
            wait (float, optional): Time to wait between retries in seconds.
                Defaults to 30.
            retries (int, optional): Number of retries. Defaults to 6.

        Raises:
            WebDriverException: If url could not be accessed.
        """
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
        """Access a url with retry.

        Args:
            url (str): The url to access.
            wait (float, optional): Time to wait asynchronously between retries in seconds.
                Defaults to 30.
            retries (int, optional): Number of retries. Defaults to 6.

        Raises:
            WebDriverException: If url could not be accessed.
        """
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

    async def sleep(self, seconds: float) -> None:
        """Sleep asynchronously and assure the window is maintained.

        Args:
            seconds (float): Seconds to sleep.
        """
        with self.maintain_window():
            await sleep_async(seconds)

    @contextmanager
    def maintain_window(self, window: str | None = None):
        """Assure the entering window is maintained on exit.

        Args:
            window (str | None, optional): A window to switch to during context.
                If None, won't switch. Defaults to None.

        Yields:
            _type_: The entering window's handle.
        """
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
        """Use own window while in context and switch back to original on exit.

        Args:
            typ ("tab" | "window", optional): Type of the new window. Defaults to "tab".
            close_after (bool, optional): Whether to close the new window on exit.
                Defaults to False.

        Yields:
            _type_: The new window's handle.
        """
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


class SeleniumExtendedChrome(Chrome, _SeleniumExtended):

    def __init__(
        self,
        headless: bool = False,
        user_agent: str | None = None,
        chrome_kwargs: dict | None = None,
    ) -> None:
        """Extended Chrome driver.

        Args:
            headless (bool, optional): Whether to run headless. Defaults to False.
            user_agent (str | None, optional): The user agent to use. Defaults to None.
            chrome_kwargs (dict[str, Any] | None, optional): Additional chromedriver
                keyword arguments. Defaults to None.
        """
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
