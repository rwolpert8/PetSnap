import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError
from deploy.upload_model import upload_and_verify, validate_endpoint


class UploadTests(unittest.TestCase):
    def test_endpoint_requires_https_without_embedded_secrets(self):
        self.assertEqual(validate_endpoint("https://t3.storageapi.dev/"), "https://t3.storageapi.dev")
        for value in ["http://example.com", "https://user:secret@example.com", "https://example.com?token=x", "https://example.com/bucket"]:
            with self.assertRaises(ValueError):
                validate_endpoint(value)

    def test_missing_object_uploads_then_verifies(self):
        client = Mock()
        client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        with patch("deploy.upload_model.fetch_model") as verify:
            upload_and_verify(client, Path("local.pth"), "bucket", "key", {})
        client.upload_file.assert_called_once_with("local.pth", "bucket", "key")
        verify.assert_called_once()

    def test_existing_object_is_not_uploaded_again(self):
        client = Mock()
        with patch("deploy.upload_model.fetch_model") as verify:
            upload_and_verify(client, Path("local.pth"), "bucket", "key", {})
        client.upload_file.assert_not_called()
        verify.assert_called_once()

    def test_access_denied_does_not_attempt_upload(self):
        client = Mock()
        client.head_object.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadObject")
        with self.assertRaises(ClientError):
            upload_and_verify(client, Path("local.pth"), "bucket", "key", {})
        client.upload_file.assert_not_called()
