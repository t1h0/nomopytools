from aiohttp import ClientSession
from lxml.etree import (
    HTML as etreeHTML,
    _Element as etreeElement,
    HTMLParser as etreeHTMLParser,
)
from lxml.html.soupparser import fromstring as lxmlsoup
from os.path import dirname as dirname


class Web:
    def __init__(self) -> None:
        pass

    @classmethod
    async def request_get(
        cls, session: ClientSession, url: str, headers: dict | None = None
    ) -> bytes | None:
        async with session.get(url, headers=headers) as response:
            return await response.content.read() if response.status < 400 else None

    @classmethod
    async def request_post(
        cls, session: ClientSession, url: str, json: dict, headers: dict | None = None
    ) -> bytes | None:
        async with session.post(url, json=json, headers=headers) as response:
            return await response.content.read() if response.status < 400 else None

    @classmethod
    def get_xp_tree(cls, source: str) -> etreeElement:
        return etreeHTML(text=source, parser=etreeHTMLParser(remove_comments=True))

    @classmethod
    def get_xp_tree_soup(cls, source: str):
        return lxmlsoup(source)