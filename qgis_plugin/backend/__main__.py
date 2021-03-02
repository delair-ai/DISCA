import argparse

from .daemon import Daemon


def parse_args():
    """parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--ssh", action="store_true", help="enable ssh connexion")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    is_ssh = args.ssh
    is_cache = not args.no_cache
    is_cpu = args.cpu
    daemon = Daemon(ssh=is_ssh, cache=is_cache, cpu=is_cpu)
    daemon.run()


if __name__ == "__main__":
    main()
