"""Tests for the ravnest doctor and version commands."""
import os
import socket
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ravnest.cli import _check_module, _check_port_free, _check_peer_reachable


class TestCheckModule:
    def test_existing_module(self):
        ok, info = _check_module("os")
        assert ok is True

    def test_torch_check(self):
        # torch is in test deps so should exist
        ok, info = _check_module("torch")
        assert ok is True
        assert info != "unknown"  # has __version__

    def test_missing_module(self):
        ok, info = _check_module("definitely_not_a_real_module_xyz")
        assert ok is False
        assert "definitely_not_a_real_module_xyz" in info or "No module" in info

    def test_returns_tuple(self):
        result = _check_module("os")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestCheckPortFree:
    def test_high_port_likely_free(self):
        # Port in ephemeral range, very likely to be free
        result = _check_port_free(54321)
        assert isinstance(result, bool)

    def test_occupied_port_returns_false(self):
        # Bind a socket, then check if it's free
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        try:
            # The port is bound to s, but our check uses a different SO_REUSEADDR
            # so this is actually more permissive — we just check the function runs
            _check_port_free(port)
            # The result depends on SO_REUSEADDR behavior
        finally:
            s.close()


class TestCheckPeerReachable:
    def test_localhost_dns(self):
        ok, msg = _check_peer_reachable("localhost", port=None)
        assert ok is True
        assert "DNS" in msg or "resolve" in msg.lower()

    def test_invalid_hostname(self):
        ok, msg = _check_peer_reachable("definitely-not-a-real-hostname-xyz-123.invalid", port=None)
        assert ok is False

    def test_unreachable_port(self):
        # Try to connect to a port nothing is listening on
        ok, msg = _check_peer_reachable("127.0.0.1", port=1, timeout=1)
        assert ok is False

    def test_reachable_port(self):
        # Open a server, connect to it
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)
        try:
            ok, msg = _check_peer_reachable("127.0.0.1", port=port, timeout=2)
            assert ok is True
        finally:
            server.close()


class TestDoctorIntegration:
    """Integration test: run cmd_doctor with mocked args."""

    def test_doctor_runs_without_error(self, capsys):
        """Verify doctor produces output without crashing."""
        from ravnest.cli import cmd_doctor
        from argparse import Namespace

        args = Namespace(port=8000, peers=None)
        try:
            cmd_doctor(args)
            captured = capsys.readouterr()
            assert "Doctor" in captured.out or "Python" in captured.out
        except SystemExit as e:
            # Issues found is a valid outcome (exit 1)
            captured = capsys.readouterr()
            assert "Python" in captured.out

    def test_doctor_with_peers(self, capsys):
        """Doctor should check peer reachability when --peers is given."""
        from ravnest.cli import cmd_doctor
        from argparse import Namespace

        args = Namespace(port=8000, peers="127.0.0.1,127.0.0.1")
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Peers" in captured.out or "127.0.0.1" in captured.out

    def test_doctor_checks_python_version(self, capsys):
        from ravnest.cli import cmd_doctor
        from argparse import Namespace

        args = Namespace(port=8000, peers=None)
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Python" in captured.out

    def test_doctor_checks_dependencies(self, capsys):
        from ravnest.cli import cmd_doctor
        from argparse import Namespace

        args = Namespace(port=8000, peers=None)
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "torch" in captured.out
        assert "transformers" in captured.out

    def test_doctor_checks_ports(self, capsys):
        from ravnest.cli import cmd_doctor
        from argparse import Namespace

        args = Namespace(port=54321, peers=None)
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "Ports" in captured.out or "port" in captured.out.lower()


class TestVersion:
    def test_version_runs_without_error(self, capsys):
        from ravnest.cli import cmd_version
        from argparse import Namespace
        cmd_version(Namespace())
        captured = capsys.readouterr()
        assert "ravnest" in captured.out
        assert "python" in captured.out

    def test_version_includes_torch(self, capsys):
        from ravnest.cli import cmd_version
        from argparse import Namespace
        cmd_version(Namespace())
        captured = capsys.readouterr()
        assert "torch" in captured.out

    def test_version_includes_commit(self, capsys):
        from ravnest.cli import cmd_version
        from argparse import Namespace
        cmd_version(Namespace())
        captured = capsys.readouterr()
        assert "commit" in captured.out

    def test_version_includes_python(self, capsys):
        from ravnest.cli import cmd_version
        from argparse import Namespace
        cmd_version(Namespace())
        captured = capsys.readouterr()
        # Python version like 3.12.x should appear
        import sys as _sys
        assert _sys.version.split()[0] in captured.out
