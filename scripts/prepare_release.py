#!/usr/bin/env python3
"""Prepare a new release."""

import re
import subprocess
from pathlib import Path
import sys


def update_version(version: str):
    """Update version in all files."""
    root = Path(__file__).parent.parent
    
    # Update src/aerotica/__init__.py
    init_file = root / "src" / "aerotica" / "__init__.py"
    content = init_file.read_text()
    content = re.sub(
        r'__version__ = ".*"',
        f'__version__ = "{version}"',
        content
    )
    init_file.write_text(content)
    
    # Update pyproject.toml
    pyproject = root / "pyproject.toml"
    content = pyproject.read_text()
    content = re.sub(
        r'version = ".*"',
        f'version = "{version}"',
        content
    )
    pyproject.write_text(content)
    
    # Update setup.cfg
    setup_cfg = root / "setup.cfg"
    content = setup_cfg.read_text()
    content = re.sub(
        r'version = .*',
        f'version = {version}',
        content
    )
    setup_cfg.write_text(content)
    
    print(f"✅ Updated version to {version}")


def update_changelog(version: str):
    """Update changelog for new version."""
    changelog = Path(__file__).parent.parent / "CHANGELOG.md"
    content = changelog.read_text()
    
    # Replace [Unreleased] with new version
    today = subprocess.check_output(
        ["date", "+%Y-%m-%d"], text=True
    ).strip()
    
    content = content.replace(
        "## [Unreleased]",
        f"## [{version}] - {today}"
    )
    
    # Add new [Unreleased] section
    new_section = f"""
## [Unreleased]

### Added
- (new features)

### Changed
- (changes)

### Fixed
- (bug fixes)

"""
    
    content = new_section + content
    changelog.write_text(content)
    
    print(f"✅ Updated CHANGELOG.md")


def git_commit(version: str):
    """Commit and tag release."""
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Release version {version}"],
        check=True
    )
    subprocess.run(["git", "tag", f"v{version}"], check=True)
    
    print(f"✅ Created git commit and tag v{version}")
    print("\nNext steps:")
    print(f"  git push origin main")
    print(f"  git push origin v{version}")
    print(f"  python -m build")
    print(f"  twine upload dist/*")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: prepare_release.py VERSION")
        sys.exit(1)
    
    version = sys.argv[1]
    
    print(f"🚀 Preparing release {version}")
    
    # Update files
    update_version(version)
    update_changelog(version)
    
    # Git operations
    response = input("\nCreate git commit and tag? (y/n): ")
    if response.lower() == 'y':
        git_commit(version)
    else:
        print("Skipping git operations")


if __name__ == "__main__":
    main()
