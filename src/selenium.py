from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
from lxml.html.soupparser import fromstring as lxmlsoup
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    element_to_be_clickable,
    invisibility_of_element_located,
    invisibility_of_element,
)
from selenium.webdriver.common.by import By
from selenium.webdriver import Firefox, Chrome
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException as SeleniumTimeout,
)
from asyncio import sleep as sleep_async
from nomotools.logger import Logger
from os.path import dirname as dirname
from time import time, sleep as sleep_sync

class _SeleniumExtended:
    def __init__(self) -> None:
        if not any((isinstance(self, Firefox), isinstance(self, Chrome))):
            raise ImportError(
                "SeleniumExtended can only act as a superclass \
                for instances of webdriver.Firefox or webdriver.Chrome. \
                To inherit from SeleniumExtended, make sure to also inherit \
                from one of tose two classes."
            )
        self.logger = Logger.get_logger(__name__, dirname(__file__))
        self.waits = {}

    def get_xp_tree(self) -> etreeElement:
        return etreeHTML(
            text=self.page_source, parser=etreeHTMLParser(remove_comments=True)
        )

    async def get_xp_tree_soup(self):
        return lxmlsoup(self.page_source)

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
        if timeout:
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
                self.logger.warning(
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
                        
class SeleniumExtendedFirefox(Firefox,_SeleniumExtended):
    
    def __init__(self, **kwargs) -> None:
        Firefox.__init__(self,**kwargs)
        _SeleniumExtended.__init__(self)

class SeleniumExtendedChrome(Chrome,_SeleniumExtended):
    
    def __init__(self, **kwargs) -> None:
        Chrome.__init__(self,**kwargs)
        _SeleniumExtended.__init__(self)
