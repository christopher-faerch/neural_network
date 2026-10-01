"""Command-line entry point for the neural-network workflows (no GUI, no Qt).

    python -m network.main <command> [arguments of that command]

    python -m network.main summary                       # model table (input shape, layers, parameters)
    python -m network.main sanity --signal filtered      # checks before training
    python -m network.main train --signal raw            # 5-seed ensemble on the fixed split
    python -m network.main evaluate --signal raw         # test metrics, confusion matrices, failure cases
    python -m network.main predict --signal raw --subject 2 --start 14.5
    python -m network.main baseline                      # feature baseline
    python -m network.main cross-validate --label v2_timing_ens
    python -m network.main compare                       # raw vs filtered tables and figures
    python -m network.main tune                          # validation-only hyperparameter check
    python -m network.main plot-examples                 # dataset figures
    python -m network.main channel-ablation --signal raw # how many of the 4 channels are needed

Every command is the `main()` of the corresponding module; arguments after the command are passed
through unchanged, so `python -m network.main train --help` shows that module's options.
Run from the project folder (the one containing network/, app/ and results/).
"""
import importlib
import sys

COMMANDS = {
    "summary": ("network.model", "print_summary"),
    "sanity": ("network.sanity_checks", "main"),
    "train": ("network.train", "main"),
    "evaluate": ("network.evaluate", "main"),
    "predict": ("network.predict", "main"),
    "baseline": ("network.baseline", "main"),
    "cross-validate": ("network.cross_validate", "main"),
    "compare": ("network.compare", "main"),
    "tune": ("network.tune", "main"),
    "plot-examples": ("network.plot_examples", "main"),
    "channel-ablation": ("network.channel_ablation", "main"),
}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["usage: python -m network.main <command> [options]", "", "commands:"]
    for name, (module, _) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  -> {module}")
    lines.append("")
    lines.append("`python -m network.main <command> --help` shows the options of that command.")
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command '{command}'\n\n{usage()}", file=sys.stderr)
        return 2
    module_name, func_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    sys.argv = [f"python -m network.main {command}"] + rest       # the module's argparse reads sys.argv
    result = getattr(module, func_name)()
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
