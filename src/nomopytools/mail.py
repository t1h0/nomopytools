from imaplib import IMAP4_SSL
from aiosmtplib import SMTP
from aiosmtplib.response import SMTPResponse
from getpass import getpass
from email import message_from_bytes
from email.message import Message, EmailMessage
from email.header import decode_header
from email.policy import SMTP as smtp_policy
import os
import logging
from dataclasses import dataclass


@dataclass(slots=True)
class Mail:
    subject: str
    sender: str
    receivers: str
    content_plain: str
    content_html: str


class Mailer:
    def __init__(
        self,
        imap_host: str,
        smtp_host: str,
        user: str,
        pw: str | None = None,
        imap_port: int = 993,
        smtp_port: int = 587,
    ) -> None:
        """The Mailer class provides methods for sending and receiving mails.

        Args:
            host (str): The host's url to connect to.
            user (str): The username to use for connection.
            pw (str | None, optional): The password to use for authentication.
                If None is given, user input will be used. Defaults to None.

        """
        self.pw = pw or getpass(
            f"Input password for IMAP and SMTP Connection of user {user}: "
        )
        self.user = user
        self.imap_host = imap_host
        self.imap_port = imap_port
        self.imap = IMAP4_SSL(host=imap_host, port=imap_port)
        self.smtp = SMTP(
            hostname=smtp_host,
            port=smtp_port,
            username=user,
            password=self.pw,
            use_tls=True,
        )

    def _imap_login(self, mailbox: str | None = None):
        server_response = None
        while True:
            match self.imap.state:
                case "LOGOUT":
                    self.imap = IMAP4_SSL(host=self.imap_host, port=self.imap_port)
                case "NONAUTH":
                    try:
                        server_response = self.imap.login(self.user, self.pw)
                    except self.imap.abort:
                        self.imap = IMAP4_SSL(host=self.imap_host, port=self.imap_port)
                case "AUTH" | "SELECTED":
                    if mailbox:
                        server_response = self.imap.select(mailbox)
                    return server_response

    def imap_search(
        self, *query: str, mailbox: str = "inbox", close_after: bool = True
    ) -> list[int]:
        """Searches for mails via imap search criteria.

        Args:
            *query (str): Imap search criteria.
            mailbox (str, optional): The mailbox to select. Defaults to "inbox".
            close_after (bool,optional): If true, closes the connection after execution.
                Defaults to True.

        Returns:
            List[int]: A list of mail identifiers that match the searched criteria.
        """
        self._imap_login(mailbox)
        result = self.imap.search(None, *query)[1][0]
        if close_after:
            self.imap.close()
            self.imap.logout()
        return [int(i) for i in result.split()] if result else []

    def imap_receive(
        self, *mail_ids: int, mailbox: str = "inbox", close_after: bool = True
    ) -> list[Message]:
        """Receive specific or the first n mails and return them
        in a list of Message objects.

        Args:
            *mail_ids (int): Integers representing identifiers of mails to receive.
                Defaults to 0 (receive all mails).
            mailbox (str, optional): The mailbox to select. Defaults to "inbox".
            close_after (bool,optional): If true, closes the connection after execution.
                Defaults to True.

        Returns:
            list[Message]: List of received mails as Message objects.
        """
        self._imap_login(mailbox)
        msg_indicies = ",".join(str(i) for i in mail_ids)
        fetch = self.imap.fetch(msg_indicies, "(RFC822)")
        if close_after:
            self.imap.close()
            self.imap.logout()
        if fetch[0] == "OK":
            received = [
                message_from_bytes(res[1]) for res in fetch[1] if isinstance(res, tuple)
            ]
        else:
            raise Exception(f"Fetching Mails was not successfull: {fetch[0]}")
        return received

    def imap_receive_recent(
        self, mailbox: str = "inbox", n: int = 0, close_after: bool = True
    ) -> list[Message]:
        """Receive specific or the first n mails and return them in a list of Message objects.

        Args:
            mailbox (str, optional): The mailbox to select. Defaults to "inbox".
            n (int, optional): Indicating the n most recent mails
                to receive. Defaults to 0 (receive all mails).
            close_after (bool,optional): If true, closes the connection after execution.
                Defaults to True.

        Returns:
            list[Message]: List of received mails as Message objects.
        """
        self._imap_login(None)
        _, msgN = self.imap.select(mailbox)
        if not msgN[0] or (msgN := int(msgN[0]) == 0):
            return []
        msg_indicies = ",".join(
            str(i)
            for i in (
                # get n most recent mails
                range(msgN, 0, -1)
                if n == 0 or n > msgN
                else range(msgN, msgN - n, -1)
            )
        )
        fetch = self.imap.fetch(msg_indicies, "(RFC822)")
        if close_after:
            self.imap.close()
            self.imap.logout()
        if fetch[0] == "OK":
            received = [
                message_from_bytes(res[1]) for res in fetch[1] if isinstance(res, tuple)
            ]
        else:
            raise Exception(f"Fetching Mails was not successfull: {fetch[0]}")
        return received

    def imap_search_and_receive(
        self,
        *query: str,
        mailbox: str = "inbox",
        download: str | None = None,
        close_after: bool = True,
    ) -> list[Mail]:
        """Search for mails via imap search criteria and receive and process
            the mails. Combination of imap_search, imap_receive and mail_read.

        Args:
            *query (str): Imap search criteria.
            mailbox (str, optional): The mailbox to select. Defaults to "inbox".
            download (str, optional): Path to download possible attachments to.
                Defaults to None.
            close_after (bool,optional): If true, closes the connection after execution.
                Defaults to True.

        Returns:
            list[Message]: List of received mails as Message objects.
        """
        self._imap_login(mailbox)
        result = self.imap.search(None, *query)[1][0]
        if not result:
            return []
        msg_indicies = ",".join(i.decode() for i in result.split())
        fetch = self.imap.fetch(msg_indicies, "(RFC822)")
        if close_after:
            self.imap.close()
            self.imap.logout()
        if fetch[0] == "OK":
            received = [
                self.mail_read(mail=message_from_bytes(res[1]), download=download)
                for res in fetch[1]
                if isinstance(res, tuple)
            ]
        else:
            raise Exception(f"Fetching Mails was not successfull: {fetch[0]}")
        return received

    def imap_delete(
        self, *mail_ids: int, mailbox: str = "inbox", close_after: bool = True
    ) -> bool:
        """Deletes a (set of) mail(s) from server.

        Args:
            mail_ids (int): The id(s) of the mail(s) to delete.
            mailbox (str, optional): The mailbox to select. Defaults to "inbox".
            close_after (bool,optional): If true, closes the connection after execution.
                Defaults to True.

        Returns:
            bool: True if delete was successful.
        """
        mail = ",".join(str(i) for i in mail_ids)
        self._imap_login(mailbox)
        self.imap.store(mail, "+FLAGS", "\\Deleted")
        if close_after:
            self.imap.close()
            self.imap.logout()
        return True

    async def smtp_send(
        self, *mails: Mail
    ) -> list[tuple[dict[str, SMTPResponse], str]]:
        """Send mails via SMTP.

        Returns:
            list[tuple[dict[str, SMTPResponse], str]]: List of one aiosmtplib response
                per mail.
        """
        email_messages = []
        for mail in mails:
            email_message = EmailMessage(smtp_policy)
            email_message.set_content(mail.content_plain)
            if mail.content_html:
                email_message.add_alternative(mail.content_html, subtype="html")
            email_message["From"] = mail.sender or self.user
            email_message["To"] = mail.receivers
            email_message["Subject"] = mail.subject
            email_messages.append(email_message)
        responses = []
        async with self.smtp:
            for message in email_messages:
                responses.append(await self.smtp.send_message(message=message))
        return responses

    def mail_read(
        self, mail: Message, download: str | None = None, shout=False
    ) -> Mail:
        """Reads Message objects and optionally downloads their attachments.

        Args:
            mail (Message): A mail as an Message object.
            download (str, optional): Path to download possible attachments to.
                Defaults to None.
            shout (bool, optional): Set to True if text/plain content should be
                printed to console. Defaults to False.

        Returns:
            Mail: The extracted mail.
        """
        result = {}
        # decode Subject, From and To
        for head in ["Subject", "From", "To"]:
            content, encoding = decode_header(mail[head])[0]
            if isinstance(content, bytes) and encoding is not None:
                # if it's a bytes, decode to str
                content = content.decode(encoding)
            result[head] = content
            if shout:
                print(f"{head}: {content}")
        result |= {"text/plain": "", "text/html": ""}
        # iterate over email parts
        for part in mail.walk():
            # extract content type of email
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            try:
                # get the email body
                body = part.get_payload(decode=True).decode()
            except:
                pass
            else:
                if (
                    content_type == "text/plain"
                    and "attachment" not in content_disposition
                ):
                    result["text/plain"] += body
                    if shout:
                        print(body)
                elif content_type == "text/html":
                    # if it's HTML, append body to HTML code
                    result["text/html"] += body
                    if shout:
                        logging.info("Mail also contains HTML code.")
            if "attachment" in content_disposition and download:
                # download attachment
                if filename := part.get_filename():
                    folder_name = "".join(
                        c if c.isalnum() else "_" for c in result["Subject"]
                    )
                    folder_path = os.path.join(download, folder_name)
                    if not os.path.isdir(folder_path):
                        # make a folder for this email (named after the subject)
                        os.mkdir(folder_path)
                    filepath = os.path.join(folder_path, filename)
                    # download attachment and save it
                    open(filepath, "wb").write(part.get_payload(decode=True))
        return Mail(
            subject=result["Subject"],
            sender=result["From"],
            receivers=result["To"],
            content_plain=result["text/plain"],
            content_html=result["text/html"],
        )
