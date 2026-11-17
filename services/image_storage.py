import hashlib
import os
import re
import time
import uuid
from typing import Optional
from urllib.parse import unquote, urlsplit

import requests
from werkzeug.utils import secure_filename

from config import IS_ONLINE


class ImageStorageError(RuntimeError):
    pass


def _cloudinary_credentials():
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()
    api_key = os.environ.get("CLOUDINARY_API_KEY", "").strip()
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", "").strip()
    if all((cloud_name, api_key, api_secret)):
        return cloud_name, api_key, api_secret

    cloudinary_url = os.environ.get("CLOUDINARY_URL", "").strip()
    if cloudinary_url:
        try:
            parsed_url = urlsplit(cloudinary_url)
            if parsed_url.scheme != "cloudinary" or not parsed_url.hostname:
                raise ValueError
            cloud_name = parsed_url.hostname
            api_key = unquote(parsed_url.username or "")
            api_secret = unquote(parsed_url.password or "")
            if not all((cloud_name, api_key, api_secret)):
                raise ValueError
            return cloud_name, api_key, api_secret
        except ValueError as exc:
            raise ImageStorageError(
                "CLOUDINARY_URL must use the format "
                "cloudinary://API_KEY:API_SECRET@CLOUD_NAME."
            ) from exc

    raise ImageStorageError(
        "Image uploads on the hosted app require either CLOUDINARY_URL "
        "or CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and "
        "CLOUDINARY_API_SECRET environment variables."
    )


def _cloudinary_upload(upload, filename: str) -> str:
    cloud_name, api_key, api_secret = _cloudinary_credentials()

    params = {
        "folder": "faq-images",
        "public_id": f"{uuid.uuid4().hex}_{os.path.splitext(filename)[0][:80]}",
        "timestamp": str(int(time.time())),
    }
    signature_source = "&".join(
        f"{key}={params[key]}" for key in sorted(params)
    )
    signature = hashlib.sha1(
        f"{signature_source}{api_secret}".encode("utf-8")
    ).hexdigest()

    try:
        response = requests.post(
            f"https://api.cloudinary.com/v1_1/{cloud_name}/image/upload",
            data={**params, "api_key": api_key, "signature": signature},
            files={
                "file": (
                    filename,
                    upload.stream,
                    upload.mimetype or "application/octet-stream",
                )
            },
            timeout=(5, 30),
        )
    except requests.RequestException as exc:
        raise ImageStorageError(
            f"Could not connect to Cloudinary ({type(exc).__name__}). "
            "Check the deployment's network access and try again."
        ) from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            response_body = response.json()
        except ValueError:
            response_body = {}
        error = response_body.get("error", {}) if isinstance(response_body, dict) else {}
        message = error.get("message", "") if isinstance(error, dict) else ""
        if not isinstance(message, str):
            message = ""
        message = re.sub(r"[\r\n\t]+", " ", message).strip()[:300]
        for credential in (api_secret, api_key):
            if credential:
                message = message.replace(credential, "[redacted]")
        detail = f": {message}" if message else ""
        raise ImageStorageError(
            f"Cloudinary rejected the upload (HTTP {response.status_code}){detail}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ImageStorageError(
            "Cloudinary returned an invalid response while storing the image."
        ) from exc

    secure_url = payload.get("secure_url") if isinstance(payload, dict) else None
    if not isinstance(secure_url, str) or not secure_url.startswith(
        "https://res.cloudinary.com/"
    ):
        raise ImageStorageError(
            "Cloudinary did not return a valid secure image URL."
        )
    return secure_url


def store_faq_image(upload, upload_folder: str) -> Optional[str]:
    """Persist an FAQ image remotely when hosted, or locally in development."""
    if not upload or not upload.filename:
        return None

    filename = secure_filename(upload.filename)
    if not filename:
        raise ImageStorageError("The uploaded image must have a valid filename.")
    if not (upload.mimetype or "").lower().startswith("image/"):
        raise ImageStorageError("Please upload a valid image file.")

    if IS_ONLINE:
        return _cloudinary_upload(upload, filename)

    os.makedirs(upload_folder, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}_{filename}"
    upload.save(os.path.join(upload_folder, stored_name))
    return f"uploads/{stored_name}"
