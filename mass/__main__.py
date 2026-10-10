import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        try:
            from .agent.cli import main as agent_main
        except ImportError as exc:
            raise SystemExit(
                "The agent needs the updated rsikit runtime and requirements-agent.txt; see docs/agent.md. "
                f"Import failed: {exc}"
            ) from None
        return agent_main(sys.argv[2:])
    from .pipeline import main as pipeline_main

    return pipeline_main()


if __name__ == "__main__":
    raise SystemExit(main())
