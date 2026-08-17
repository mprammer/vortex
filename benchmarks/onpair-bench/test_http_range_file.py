#!/usr/bin/env python3
"""Deterministic contract tests for run.py::_HttpRangeFile."""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("onpair_run", BENCH / "run.py")
assert spec and spec.loader
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class Response:
    def __init__(self, *, status=200, headers=None, url="https://cdn/object", data=b""):
        self.status = status
        self.headers = headers or {}
        self._url = url
        self._data = data
        self.read_calls = 0
        self.closed = False

    def geturl(self):
        return self._url

    def read(self, n=-1):
        self.read_calls += 1
        return self._data if n < 0 else self._data[:n]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True
        return False


class ScriptedOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def head(size=8, *, etag='"object-a"', headers=None, **kwargs):
    response_headers = {"Content-Length": str(size), "ETag": etag}
    response_headers.update(headers or {})
    return Response(headers=response_headers, **kwargs)


def byte_range(start, end, total, data, *, etag='"object-a"', headers=None, **kwargs):
    response_headers = {
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(end - start + 1),
        "ETag": etag,
    }
    response_headers.update(headers or {})
    return Response(status=206, headers=response_headers, data=data, **kwargs)


def request_headers(request):
    return {name.lower(): value for name, value in request.header_items()}


class HttpRangeFileTests(unittest.TestCase):
    def make(self, *outcomes, max_attempts=4):
        scripted = ScriptedOpener(outcomes)
        file = run._HttpRangeFile(
            "https://origin/object",
            {"Authorization": "Bearer secret"},
            opener=scripted,
            sleep=lambda _delay: None,
            max_attempts=max_attempts,
        )
        return file, scripted

    def test_exact_ranges_are_validated_and_eof_does_not_request(self):
        file, calls = self.make(
            head(5),
            byte_range(0, 3, 5, b"abcd"),
            byte_range(4, 4, 5, b"e"),
        )
        first = bytearray(4)
        self.assertEqual(file.readinto(first), 4)
        self.assertEqual(bytes(first), b"abcd")
        self.assertEqual(file.tell(), 4)

        second = bytearray(4)
        self.assertEqual(file.readinto(second), 1)
        self.assertEqual(bytes(second[:1]), b"e")
        self.assertEqual(file.tell(), 5)

        before = len(calls.requests)
        self.assertEqual(file.readinto(bytearray(1)), 0)
        self.assertEqual(len(calls.requests), before)

        request, timeout = calls.requests[1]
        self.assertEqual(request.full_url, "https://origin/object")
        self.assertEqual(request_headers(request)["range"], "bytes=0-3")
        self.assertEqual(request_headers(request)["if-match"], '"object-a"')
        self.assertEqual(timeout, run.HTTP_TIMEOUT_SECONDS)

    def test_zero_length_object_and_buffer_do_not_issue_ranges(self):
        file, calls = self.make(head(0))
        self.assertEqual(file.read(), b"")
        self.assertEqual(file.readinto(bytearray()), 0)
        self.assertEqual(file.tell(), 0)
        self.assertEqual(len(calls.requests), 1)

    def test_later_200_is_rejected_before_body_is_read(self):
        later = Response(status=200, data=b"whole-object")
        file, _ = self.make(head(8), byte_range(0, 3, 8, b"abcd"), later)
        self.assertEqual(file.readinto(bytearray(4)), 4)
        with self.assertRaisesRegex(OSError, "ignored a Range request"):
            file.readinto(bytearray(4))
        self.assertEqual(later.read_calls, 0)
        self.assertTrue(later.closed)
        self.assertEqual(file.tell(), 4)

    def test_wrong_content_range_or_etag_is_rejected(self):
        cases = [
            Response(
                status=206,
                headers={
                    "Content-Range": "bytes 9-12/13",
                    "Content-Length": "4",
                    "ETag": '"object-a"',
                },
                data=b"EVIL",
            ),
            byte_range(0, 3, 4, b"EVIL", etag='"object-b"'),
        ]
        for response in cases:
            with self.subTest(headers=response.headers):
                file, _ = self.make(head(4), response, max_attempts=1)
                with self.assertRaises(OSError):
                    file.readinto(bytearray(4))
                self.assertEqual(response.read_calls, 0)
                self.assertTrue(response.closed)
                self.assertEqual(file.tell(), 0)

    def test_encoded_range_is_rejected_before_body_is_read(self):
        encoded = byte_range(
            0,
            3,
            4,
            b"abcd",
            headers={"Content-Encoding": "gzip"},
        )
        file, _ = self.make(head(4), encoded, max_attempts=1)
        with self.assertRaisesRegex(OSError, "encoded a byte-range response"):
            file.readinto(bytearray(4))
        self.assertEqual(encoded.read_calls, 0)
        self.assertTrue(encoded.closed)
        self.assertEqual(file.tell(), 0)

    def test_short_body_is_discarded_and_retried_from_same_offset(self):
        short = byte_range(0, 3, 4, b"ab")
        correct = byte_range(0, 3, 4, b"abcd")
        file, calls = self.make(head(4), short, correct)
        output = bytearray(4)
        self.assertEqual(file.readinto(output), 4)
        self.assertEqual(bytes(output), b"abcd")
        self.assertEqual(file.tell(), 4)
        self.assertTrue(short.closed)
        self.assertEqual(
            [request_headers(request)["range"] for request, _ in calls.requests[1:]],
            ["bytes=0-3", "bytes=0-3"],
        )

    def test_oversized_body_is_never_committed(self):
        responses = [byte_range(0, 3, 4, b"abcde") for _ in range(2)]
        file, _ = self.make(head(4), *responses, max_attempts=2)
        with self.assertRaisesRegex(OSError, "returned 5 bytes"):
            file.readinto(bytearray(4))
        self.assertEqual(file.tell(), 0)
        self.assertTrue(all(response.closed for response in responses))

    def test_connection_reset_and_429_are_retried(self):
        rate_limit_body = io.BytesIO(b"rate limited")
        rate_limit = urllib.error.HTTPError(
            "https://cdn/object", 429, "rate limited", {}, rate_limit_body
        )
        file, calls = self.make(
            head(4),
            ConnectionResetError("reset"),
            rate_limit,
            byte_range(0, 3, 4, b"abcd"),
        )
        self.assertEqual(file.read(), b"abcd")
        self.assertEqual(len(calls.requests), 4)
        self.assertTrue(rate_limit_body.closed)

    def test_retry_exhaustion_does_not_advance_position(self):
        file, calls = self.make(
            head(4),
            ConnectionResetError("first"),
            ConnectionResetError("second"),
            max_attempts=2,
        )
        with self.assertRaises(ConnectionResetError):
            file.readinto(bytearray(4))
        self.assertEqual(file.tell(), 0)
        self.assertEqual(len(calls.requests), 3)

    def test_416_is_not_retried_and_does_not_advance_position(self):
        body = io.BytesIO(b"range unsatisfiable")
        error = urllib.error.HTTPError(
            "https://cdn/object",
            416,
            "range unsatisfiable",
            {},
            body,
        )
        file, calls = self.make(head(4), error)
        with self.assertRaises(urllib.error.HTTPError):
            file.readinto(bytearray(4))
        self.assertEqual(file.tell(), 0)
        self.assertEqual(len(calls.requests), 2)
        self.assertTrue(body.closed)

    def test_head_requires_length_and_strong_etag(self):
        for response in [
            Response(headers={"ETag": '"object-a"'}),
            Response(headers={"Content-Length": "4"}),
            Response(headers={"Content-Length": "4", "ETag": 'W/"object-a"'}),
        ]:
            with self.subTest(headers=response.headers):
                scripted = ScriptedOpener([response])
                with self.assertRaises(OSError):
                    run._HttpRangeFile(
                        "https://origin/object",
                        opener=scripted,
                        sleep=lambda _delay: None,
                    )
                self.assertTrue(response.closed)

    def test_seek_modes_and_read_only_buffer(self):
        file, calls = self.make(head(8))
        self.assertEqual(file.seek(3), 3)
        self.assertEqual(file.seek(2, io.SEEK_CUR), 5)
        self.assertEqual(file.seek(-1, io.SEEK_END), 7)
        self.assertEqual(file.seek(5, io.SEEK_END), 13)
        self.assertEqual(file.readinto(bytearray(1)), 0)
        with self.assertRaises(ValueError):
            file.seek(-9)
        with self.assertRaises(ValueError):
            file.seek(0, 99)
        file.seek(0)
        with self.assertRaises(TypeError):
            file.readinto(memoryview(b"read-only"))
        self.assertEqual(len(calls.requests), 1)


class AuthorizationRedirectTests(unittest.TestCase):
    def redirect(self, target):
        handler = run._ScopedAuthorizationRedirectHandler()
        request = urllib.request.Request(
            "https://huggingface.co/datasets/repo/resolve/rev/file",
            headers={"Authorization": "Bearer secret", "Range": "bytes=0-3"},
        )
        return handler.redirect_request(request, None, 302, "Found", {}, target)

    def test_authorization_is_removed_on_cross_origin_redirect(self):
        redirected = self.redirect("https://cdn.example/object")
        headers = request_headers(redirected)
        self.assertNotIn("authorization", headers)
        self.assertEqual(headers["range"], "bytes=0-3")

    def test_authorization_is_retained_on_same_origin_redirect(self):
        redirected = self.redirect("https://huggingface.co/other")
        self.assertEqual(request_headers(redirected)["authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main(verbosity=2)
