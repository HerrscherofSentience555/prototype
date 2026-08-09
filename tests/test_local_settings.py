from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.local_settings import (  # noqa: E402
    EXAMPLE_SETTINGS_PATH,
    LOCAL_SETTINGS_PATH,
    load_local_settings,
    write_local_settings_template,
)


class LocalSettingsTests(unittest.TestCase):
    def test_loads_example_when_local_settings_is_absent(self) -> None:
        settings = load_local_settings()

        expected_source = LOCAL_SETTINGS_PATH if LOCAL_SETTINGS_PATH.exists() else EXAMPLE_SETTINGS_PATH
        self.assertEqual(settings.source, str(expected_source))
        self.assertIn(settings.runtime.platform, {"windows_wsl", "linux_native"})
        self.assertTrue(settings.paths.default_model_file)

    def test_write_local_settings_template_copies_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "local_settings.yaml"

            result = write_local_settings_template(target)

            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), EXAMPLE_SETTINGS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
