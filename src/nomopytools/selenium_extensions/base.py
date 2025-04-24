# third-party imports
from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
from bs4 import BeautifulSoup
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    D,
    element_to_be_clickable,
    invisibility_of_element_located,
    invisibility_of_element,
    visibility_of_element_located,
)
from selenium.webdriver.common.by import By
from selenium.webdriver import (
    Firefox,
    Chrome,
)
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.shadowroot import ShadowRoot
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException as SeleniumTimeout,
    NoSuchElementException,
)

# built-in imports
from contextlib import contextmanager, asynccontextmanager
from asyncio import sleep as sleep_async
from loguru import logger
from os.path import dirname as dirname
from time import time, sleep as sleep_sync
from typing import Literal, Callable, overload, TypeVar
import warnings
from random import gauss, expovariate

T = TypeVar("T")


class _SeleniumExtended:

    _MEAN_REACTION_TIME = 1.5
    """Mean reaction time for humanoid reaction behavior in seconds."""
    _MINIMUM_TYPING_DELAY = 0.035
    """Minimum delay between typing characters for humanoid typing behavior in seconds."""

    def __init__(self) -> None:
        """Extended Selenium Driver Superclass."""
        if not (isinstance(self, Firefox | Chrome)):
            raise ImportError(
                "SeleniumExtended can only act as a superclass"
                " for instances of webdriver.Firefox or webdriver.Chrome."
                " To inherit from _SeleniumExtended, make sure to also inherit"
                " from one of those two classes."
            )
        self.last_interaction: float = 0
        """Timestamp of the last interaction with the browser"""

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

    @overload
    def get_elem_xp(
        self,
        xpath: str,
        timeout: int = 10,
        method: Callable[
            ..., Callable[[D], Literal[False] | WebElement]
        ] = element_to_be_clickable,
    ) -> "ExtendedWebElement": ...

    @overload
    def get_elem_xp(
        self,
        xpath: str,
        timeout: int = 10,
        method: Callable[
            ..., Callable[[D], Literal[False] | T]
        ] = element_to_be_clickable,
    ) -> T: ...

    def get_elem_xp(
        self,
        xpath: str,
        timeout: int = 10,
        method: Callable[
            ..., Callable[[D], Literal[False] | T]
        ] = element_to_be_clickable,
    ) -> "T | ExtendedWebElement":
        """Get an element using xpath.

        Args:
            xpath (str): The xpath to use.
            timeout (int, optional): Timeout for waiting. Defaults to 10.

        Returns:
            WebElement: The element.
        """
        out = WebDriverWait(self, timeout).until(method((By.XPATH, xpath)))

        return ExtendedWebElement(out) if isinstance(out, WebElement) else out

    def get_elem_xp_humanoid(
        self,
        xpath: str,
        timeout: int = 10,
    ) -> "ExtendedWebElement":
        return self.get_elem_xp(
            xpath=xpath, timeout=timeout, method=visibility_of_element_located
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
    ) -> "list[ExtendedWebElement]": ...

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[False],
        timeout: None = None,
    ) -> "ExtendedWebElement | None": ...

    @overload
    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: Literal[False],
        timeout: int,
    ) -> "ExtendedWebElement": ...

    async def get_shadowed_elem(
        self,
        shadow_root_selector: str,
        elem_css_selector: str,
        get_all: bool = True,
        timeout: int | None = None,
    ) -> "list[ExtendedWebElement] | ExtendedWebElement | None":
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
            list[ExtendedWebElement] | ExtendedWebElement | None: The matched element(s)
                or None if None was found.
        """
        shadow_root = await self.get_elem_js(
            selector=shadow_root_selector, timeout=timeout, get_all=False
        )

        if not shadow_root:
            return None

        shadow_root = shadow_root.shadow_root

        out = await self.timeout(
            func=lambda: self.search_shadow_root(
                shadow_root=shadow_root,
                elem_css_selector=elem_css_selector,
                get_all=get_all,
            ),
            timeout=timeout,
        )

        if isinstance(out, list):
            return [ExtendedWebElement(elem) for elem in out]
        elif isinstance(out, WebElement):
            return ExtendedWebElement(out)
        return out

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
    def wait_for_url_change(self, timeout: int = 10):
        """Wait for url change after context.

        Args:
            timeout (int, optional): Timeout for waiting. Defaults to 10.


        Yields:
            str: The entering window's url.
        """
        current_url = self.current_url

        try:
            yield current_url
        finally:
            WebDriverWait(self, timeout).until(
                lambda driver: driver.current_url != current_url
            )

    @contextmanager
    def maintain_window(self, window: str | None = None):
        """Assure the entering window is maintained on exit.

        Args:
            window (str | None, optional): A window to switch to during context.
                If None, won't switch. Defaults to None.

        Yields:
            str: The entering window's handle.
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
            str: The new window's handle.
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

    @asynccontextmanager
    async def react(self):
        """Simulates human reaction time around an interaction."""
        # pre-reaction

        # create a random interaction, gaussian distributed around the mean reaction time
        await self.sleep(abs(gauss(self._MEAN_REACTION_TIME, 0.5)))

        try:
            # action
            yield
        finally:
            # post-reaction
            # save the time after the yielded interaction
            self.last_interaction = time()


class ExtendedWebElement(WebElement):

    def __init__(self, web_element: WebElement) -> None:
        """WebElement with extended functionality.

        Args:
            web_element (WebElement): The WebElement to convert.
        """
        super().__init__(web_element.parent, web_element.id)

    @property
    def parent(self) -> _SeleniumExtended:
        return super().parent

    async def click_humanoid(self) -> None:
        async with self.parent.react():
            return super().click()

    def send_keys_humanoid(self, *value: str) -> None:
        for val in value:
            for key in val:
                super().send_keys(key)
                # simulate typing delay
                # we add a random delay from the exponential distribution
                # with mean = 0.1 and lambda = 1/mean
                sleep_sync(self.parent._MINIMUM_TYPING_DELAY + expovariate(1 / 0.015))

    def select(
        self,
        value: str | None = None,
        visible_text: str | None = None,
        index: int | None = None,
    ) -> None:
        """Select an option of the element (self needs to be a select element)
        by value, visible text or index. Exactly one of value, visible_text or index
        must be set.

        Args:
            value (str | None, optional): Value to select. Defaults to None.
            visible_text (str | None, optional): Visible text to select. Defaults to None.
            index (int | None, optional): Index to select. Defaults to None.

        Raises:
            ValueError: If not exactly one of value, visible_text or index is set.
        """
        if sum(i is not None for i in (value, visible_text, index)) != 1:
            raise ValueError("Exactly one of value, visible_text, index must be set.")

        if value is not None:
            return Select(self).select_by_value(value)
        if visible_text is not None:
            return Select(self).select_by_visible_text(visible_text)
        if index is not None:
            return Select(self).select_by_index(index)
