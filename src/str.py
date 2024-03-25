from re import sub


def to_var(string: str) -> str:
    """Converts a string to be a valid python variable name.

    Args:
        string (str): The string to convert.

    Returns:
        str: The string as a valid python variable name.
    """
    return sub(r"\W|^(?=\d)", "_", string)
