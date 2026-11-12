"""Use the same supported entry point as the installed pas command."""

from pas_release.cli import main

raise SystemExit(main())
