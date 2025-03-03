# third-party imports
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.fernet import Fernet, InvalidToken

# built-in imports
from getpass import getpass
from os import urandom
from typing import Any, overload
from base64 import urlsafe_b64encode

# local imports
from ..console import delete_lines
from .types import HasherPresets, PasswordProperties


class PasswordManager:

    # official argon2 parameter recommendations from rfc9106
    RFC9106: dict[str, HasherPresets] = {
        "high_memory": {
            "length": 32,
            "iterations": 1,
            "lanes": 4,
            "memory_cost": 2**21,
        },
        "low_memory": {
            "length": 32,
            "iterations": 3,
            "lanes": 4,
            "memory_cost": 2**16,
        },
    }

    _VERIFIER_PHRASE = b"NomoPyToolsPasswordManager"

    def __init__(self, hasher_presets: HasherPresets = RFC9106["low_memory"]) -> None:
        """Handle password operations.

        Args:
            hasher_presets (HasherPresets, optional): Presets for password hashing.
                Defaults to RFC9106["low_memory"].
        """
        self._master_password = None
        self._fernet = None

        self.hasher_presets = hasher_presets

    @property
    def master_password(self) -> bytes:
        if self._master_password is None:
            raise AttributeError("Master password not set.")
        return self._master_password

    @master_password.setter
    def master_password(self, _: Any) -> None:
        raise AttributeError(
            "Direct master password setting not allowed." " Use set_master_password()."
        )

    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            raise AttributeError("Master password not set.")
        return self._fernet

    @fernet.setter
    def fernet(self, _: Any) -> None:
        raise AttributeError(
            "Direct fernet setting not allowed."
            " Use set_master_password() or get_master_password()."
        )

    def get_master_password(
        self,
        prompt: str = "Enter your master password",
        confirm_prompt: str = "Confirm master password",
    ) -> PasswordProperties:
        """Get the master password from user input.

        Args:
            prompt (str, optional): The prompt to ask for a password.
                Defaults to "Enter your master password".
            confirm_prompt (str, optional): The prompt to ask for password confirmation.
                Defaults to "Confirm master password".

        Returns:
            PasswordProperties: Generated salt and verifier.
        """
        return self.set_master_password(
            self.get_password(
                prompt=prompt,
                confirm_prompt=confirm_prompt,
            )
        )

    def verify_master_password(
        self, salt: bytes, verifier: bytes, tries: int = 3
    ) -> None:
        """Verifies master password from user input and sets it if correct.

        Args:
            salt (bytes, optional): The salt to use. If None, will generate random salt
                Defaults to None.
            verifier (bytes, optional): The verifier to use. If None, will generate a
                verifier. Defaults to None.
            tries (int, optional): Tries to give the user to input the right password.
                Defaults to 3.
        """
        if tries <= 0:
            raise ValueError("tries must be greater than 0.")

        for ntry in range(tries):
            try:
                self.set_master_password(
                    pw=self.get_password("Enter your master password", None),
                    salt=salt,
                    verifier=verifier,
                )
                return
            except ValueError as e:
                if ntry == tries - 1:
                    raise ValueError("Wrong password too many times!") from e
                input(f"Wrong password! {tries-ntry-1} tries left.")
                delete_lines(1)

    @overload
    def set_master_password(
        self, pw: str, salt: None = None, verifier: None = None
    ) -> PasswordProperties: ...

    @overload
    def set_master_password(self, pw: str, salt: bytes, verifier: bytes) -> None: ...

    def set_master_password(
        self, pw: str, salt: bytes | None = None, verifier: bytes | None = None
    ) -> PasswordProperties | None:
        """Set the master password manually.

        Args:
            pw (str | None, optional): The new master password.
            salt (bytes, optional): The salt to use. If None, will generate random salt.
                Defaults to None.
            verifier (bytes, optional): The verifier to use. If None, will generate
                verifier. Defaults to None.

        Returns:
            PasswordProperties | None: Generated salt and verifier if not provided as
                arguments.
        """
        if salt is None or verifier is None:
            # we need both salt and verifier
            salt = self.generate_salt()
            verifier = None
            out = True
        else:
            # don't output salt and verifier if provided
            out = False

        hashed_pw = self.hash_password(
            pw,
            presets=self.hasher_presets,
            salt=salt,
        )

        fernet = Fernet(urlsafe_b64encode(hashed_pw))

        verifier = verifier or fernet.encrypt(self._VERIFIER_PHRASE)

        try:
            # test if master password fits verifier
            fernet.decrypt(verifier)
        except InvalidToken as e:
            raise ValueError("Wrong password!") from e

        # Password is correct, set the attributes

        super().__setattr__("_master_password", hashed_pw)

        super().__setattr__("_fernet", fernet)

        if out:
            return PasswordProperties(salt=salt, verifier=verifier)

    def encrypt(self, data: str | bytes) -> bytes:
        """Encrypt data using Fernet.

        Args:
            data (bytes): The data to encrypt.

        Returns:
            bytes: The encrypted data.
        """
        return self.fernet.encrypt(
            data.encode("utf-8") if isinstance(data, str) else data
        )

    def decrypt(self, data: str | bytes) -> bytes:
        """Decrypt data using Fernet.

        Args:
            data (bytes): The data to decrypt.

        Returns:
            bytes: The decrypted data.
        """
        return self.fernet.decrypt(
            data.encode("utf-8") if isinstance(data, str) else data
        )

    @classmethod
    def get_hasher(
        cls, presets: HasherPresets | None = None, salt: bytes | None = None
    ) -> Argon2id:
        """Get an Argon2id hasher to hash passwords.

        Args:
            preset (HasherPresets | None, optional): Presets to use.
                If None, will use RFC9106["low_memory"].
            salt (bytes | None, optional): The salt to use. If None, will generate
                a random salt using generate_salt() default. Defaults to None.

        Returns:
            Argon2id: The hasher.
        """
        return Argon2id(
            salt=salt or urandom(16),
            **(presets or cls.RFC9106["low_memory"]),
            ad=None,
            secret=None,
        )

    @classmethod
    def hash_password(
        cls,
        password: str,
        presets: HasherPresets | None = None,
        salt: bytes | None = None,
    ) -> bytes:
        """Hash a password.

        Args:
            password (str): The password to hash.
            presets (HasherPresets | NOne, optional): Presets for the Argon2id hasher.
                If None, will use get_hasher() default. Defaults to None.
            salt (bytes | None, optional): Salt to use. If None, will use get_hasher()
                default. Defaults to None.


        Returns:
            bytes: The hashed password.
        """
        return cls.get_hasher(presets, salt).derive(password.encode("utf-8"))

    @classmethod
    def get_password(
        cls,
        prompt: str = "Enter your password",
        confirm_prompt: str | None = "Confirm password",
    ) -> str:
        """Get a password from user input.

        Args:
            prompt (str, optional): The prompt to ask for a password.
                Defaults to "Enter your password".
            confirm_prompt (str, optional): The prompt to ask for password confirmation.
                If None, won't ask for confirmation. Defaults to "Confirm password".

        Returns:
            str: The password.
        """
        prompt_len = 1 + prompt.count("\n")

        if confirm_prompt is not None:
            prompt_len += 1 + confirm_prompt.count("\n")

        while True:
            pw = getpass(f"{prompt}: ")

            if confirm_prompt is not None and pw != getpass(f"{confirm_prompt}: "):
                input("Passwords do not match! Press any key to try again.")
                delete_lines(1 + prompt_len)
                continue

            delete_lines(prompt_len)
            return pw

    @classmethod
    def generate_salt(cls, n: int = 16) -> bytes:
        """Generates a random salt of size n.

        Args:
            n (int, optional): The size of the salt in bits. Defaults to 16.

        Returns:
            bytes: The salt.
        """
        return urandom(n)
