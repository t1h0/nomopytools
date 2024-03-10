def endswith(substring: str, string: str = "text()") -> str:
    """Return an XPATH 1.0 equivalent to the endswith function.

    Args:
        substring (str): Substring that the string should end with.
        string (str, optional): The string to check. Defaults to "text()".

    Returns:
        str: The resulting xpath.
    """
    return f"'{substring}' = substring({string},string-length({string}) - string-length('{substring}')+1)"