import argparse
import logging
import sys

from .commands.swap import register_swap_command
from .version import __version__

logging.basicConfig(
    level=logging.INFO,
    format="[TinyFace %(levelname)s]: %(message)s",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"TinyFace v{__version__} by https://del.wang",
        prog="tinyface",
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    register_swap_command(subparsers)

    return parser


def _main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def main() -> int:
    try:
        _main()
        return 0
    except KeyboardInterrupt:
        return 1
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
