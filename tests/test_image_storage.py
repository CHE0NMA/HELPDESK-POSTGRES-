import hashlib
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from werkzeug.datastructures import FileStorage

from services.formatting import format_bot_message
from services.image_storage import ImageStorageError, store_faq_image


class ImageStorageTests(unittest.TestCase):
    def setUp(self):
        self.upload_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.upload_dir.cleanup)

    def test_local_upload_creates_missing_folder_and_sanitizes_filename(self):
        upload = FileStorage(
            stream=io.BytesIO(b"image-bytes"),
            filename="Admissions map.png",
            content_type="image/png",
        )
        folder = os.path.join(self.upload_dir.name, "new", "uploads")

        image_path = store_faq_image(upload, folder)

        if image_path is None:
            self.fail("An uploaded image should return its stored path.")
        self.assertTrue(image_path.startswith("uploads/"))
        with open(os.path.join(folder, image_path.split("/", 1)[1]), "rb") as saved:
            self.assertEqual(saved.read(), b"image-bytes")

    def test_online_upload_uses_signed_cloudinary_request(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions map.png",
            content_type="image/png",
        )
        response = Mock()
        response.json.return_value = {
            "secure_url": "https://res.cloudinary.com/demo/image/upload/faq.png",
        }
        with patch("services.image_storage.IS_ONLINE", True), \
                patch.dict(os.environ, {
                    "CLOUDINARY_CLOUD_NAME": "demo",
                    "CLOUDINARY_API_KEY": "public-key",
                    "CLOUDINARY_API_SECRET": "private-secret",
                }), \
                patch("services.image_storage.time.time", return_value=1234), \
                patch("services.image_storage.requests.post", return_value=response) as post:
            image_url = store_faq_image(upload, self.upload_dir.name)

        self.assertEqual(
            image_url,
            "https://res.cloudinary.com/demo/image/upload/faq.png",
        )
        self.assertEqual(
            post.call_args.args[0],
            "https://api.cloudinary.com/v1_1/demo/image/upload",
        )
        data = post.call_args.kwargs["data"]
        signature_source = "&".join(
            f"{key}={data[key]}"
            for key in sorted(("folder", "public_id", "timestamp"))
        )
        expected_signature = hashlib.sha1(
            f"{signature_source}private-secret".encode()
        ).hexdigest()
        self.assertEqual(data["signature"], expected_signature)
        self.assertEqual(data["api_key"], "public-key")

    def test_online_upload_accepts_cloudinary_url(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions.png",
            content_type="image/png",
        )
        response = Mock()
        response.json.return_value = {
            "secure_url": "https://res.cloudinary.com/demo/image/upload/faq.png",
        }
        with patch("services.image_storage.IS_ONLINE", True), \
                patch.dict(os.environ, {
                    "CLOUDINARY_URL": "cloudinary://public-key:private-secret@demo",
                }, clear=True), \
                patch("services.image_storage.requests.post", return_value=response) as post:
            image_url = store_faq_image(upload, self.upload_dir.name)

        self.assertIn("res.cloudinary.com/demo", image_url)
        self.assertEqual(post.call_args.kwargs["data"]["api_key"], "public-key")
        self.assertEqual(
            post.call_args.args[0],
            "https://api.cloudinary.com/v1_1/demo/image/upload",
        )

    def test_online_upload_rejects_malformed_cloudinary_url(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions.png",
            content_type="image/png",
        )
        with patch("services.image_storage.IS_ONLINE", True), \
                patch.dict(os.environ, {"CLOUDINARY_URL": "cloudinary"}, clear=True):
            with self.assertRaisesRegex(ImageStorageError, "CLOUDINARY_URL"):
                store_faq_image(upload, self.upload_dir.name)

    def test_online_upload_rejects_non_cloudinary_secure_url(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions.png",
            content_type="image/png",
        )
        response = Mock()
        response.json.return_value = {"secure_url": "https://example.com/image.png"}
        with patch("services.image_storage.IS_ONLINE", True), \
                patch.dict(os.environ, {
                    "CLOUDINARY_CLOUD_NAME": "demo",
                    "CLOUDINARY_API_KEY": "public-key",
                    "CLOUDINARY_API_SECRET": "private-secret",
                }), \
                patch("services.image_storage.requests.post", return_value=response):
            with self.assertRaisesRegex(
                ImageStorageError,
                "valid secure image URL",
            ):
                store_faq_image(upload, self.upload_dir.name)

    def test_online_upload_reports_cloudinary_request_errors(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions.png",
            content_type="image/png",
        )
        with patch("services.image_storage.IS_ONLINE", True), \
                patch.dict(os.environ, {
                    "CLOUDINARY_CLOUD_NAME": "demo",
                    "CLOUDINARY_API_KEY": "public-key",
                    "CLOUDINARY_API_SECRET": "private-secret",
                }), \
                patch(
                    "services.image_storage.requests.post",
                    side_effect=requests.Timeout("timed out"),
                ):
            with self.assertRaisesRegex(
                ImageStorageError,
                "Cloudinary could not store the image",
            ):
                store_faq_image(upload, self.upload_dir.name)

    def test_online_upload_without_credentials_fails_with_setup_message(self):
        upload = FileStorage(
            stream=io.BytesIO(b"cloud-image"),
            filename="Admissions.png",
            content_type="image/png",
        )
        env = {
            key: os.environ.pop(key, None)
            for key in (
                "CLOUDINARY_CLOUD_NAME",
                "CLOUDINARY_API_KEY",
                "CLOUDINARY_API_SECRET",
                "CLOUDINARY_URL",
            )
        }
        try:
            with patch("services.image_storage.IS_ONLINE", True):
                with self.assertRaisesRegex(
                    ImageStorageError,
                    "CLOUDINARY_CLOUD_NAME",
                ):
                    store_faq_image(upload, self.upload_dir.name)
        finally:
            for key, value in env.items():
                if value is not None:
                    os.environ[key] = value

    def test_chat_formatter_renders_cloudinary_and_local_upload_urls(self):
        cloud_url = "https://res.cloudinary.com/demo/image/upload/faq.png"

        self.assertIn(f'src="{cloud_url}"', format_bot_message(f"[Image: {cloud_url}]"))
        self.assertIn(
            'src="/uploads/local-image.png"',
            format_bot_message("[Image: uploads/local-image.png]"),
        )
        self.assertNotIn(
            "<img",
            format_bot_message("[Image: javascript:alert(1)]"),
        )


if __name__ == "__main__":
    unittest.main()
