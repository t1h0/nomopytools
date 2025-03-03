from sys import stdout

def delete_lines(n:int = 1) -> None:
    """Deletes n previous lines.

    Args:
        n (int, optional): Number of lines to delete. Defaults to 1.
    """
    stdout.write("\033[F\033[K" * n)