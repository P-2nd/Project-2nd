"""Tests for platform-specific environment setup behavior."""

import unittest

from src.environment.setup_python312_env import environment_profile, required_packages, requires_macos_openmp


class EnvironmentSetupTests(unittest.TestCase):
    def test_environment_profile_for_supported_platforms(self) -> None:
        self.assertEqual(environment_profile("darwin", "x86_64"), "Intel macOS")
        self.assertEqual(environment_profile("darwin", "arm64"), "Apple Silicon macOS")
        self.assertEqual(environment_profile("win32", "AMD64"), "Windows")

    def test_intel_macos_requires_binary_compatible_numba_stack(self) -> None:
        packages = required_packages("Intel macOS")

        self.assertEqual(packages["numba"], "numba")
        self.assertEqual(packages["llvmlite"], "llvmlite")

    def test_other_platforms_do_not_add_intel_only_packages(self) -> None:
        packages = required_packages("Apple Silicon macOS")

        self.assertNotIn("numba", packages)
        self.assertNotIn("llvmlite", packages)

    def test_openmp_is_required_only_on_macos_profiles(self) -> None:
        self.assertTrue(requires_macos_openmp("Intel macOS"))
        self.assertTrue(requires_macos_openmp("Apple Silicon macOS"))
        self.assertFalse(requires_macos_openmp("Windows"))
