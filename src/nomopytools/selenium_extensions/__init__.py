from .firefox import ExtendedFirefox as Firefox
from .chrome import ExtendedChrome as Chrome

ExtendedSeleniumDriver = Chrome | Firefox
