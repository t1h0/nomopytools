from typing import Callable, TypeVar, ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def copy_doc(doc_source: Callable[P, T]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to copy the docstring of doc_source to another.
    Inspired by Trevor (stackoverflow.com/users/13905088/trevor)
    from: stackoverflow.com/questions/68901049/
        copying-the-docstring-of-function-onto-another-function-by-name

    Args:
        doc_source (Callable): The source function to copy the docstring from.

    Returns:
        Callable: The decorated function.

    """

    def wrapped(doc_target: Callable[P, T]) -> Callable[P, T]:
        doc_target.__doc__ = doc_source.__doc__
        return doc_target

    return wrapped
