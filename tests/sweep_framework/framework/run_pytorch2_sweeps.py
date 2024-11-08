import os
import sys
import glob
import subprocess
import sqlite3
from pathlib import Path
from tabulate import tabulate
import argparse


def find_pytorch2_files(parent_folder):
    pattern = os.path.join(parent_folder, "**", "*pytorch2.py")
    return glob.glob(pattern, recursive=True)


def get_module_name(file_path, sweeps_dir):
    # Extract the module name relative to the sweeps directory
    relative_path = os.path.relpath(file_path, sweeps_dir)
    module_name = relative_path.replace(".py", "").replace(os.sep, ".")

    return module_name


def run_commands(module_name, args):
    commands = [
        f"python3 tests/sweep_framework/sweeps_parameter_generator.py --elastic {args.elastic} --module-name {module_name}",
        f"python3 tests/sweep_framework/sweeps_runner.py --elastic {args.elastic} --module-name {module_name} --suite-name {args.suite_name}",
        f"python3 tests/sweep_framework/framework/export_to_sqlite.py --elastic {args.elastic} --dump_path {args.dump_path} --filter-string {module_name}",
    ]
    for cmd in commands:
        print(f"Running command: {cmd}")
        attempts = 0
        success = False
        while attempts < 3 and not success:
            attempts = attempts + 1
            try:
                subprocess.run(cmd, shell=True, check=True)
                success = True
            except subprocess.CalledProcessError as e:
                print(f"Command failed with {e}. Attempt {attempts} of 3.")
                if attempts == 3:
                    print(f"Unable to process {cmd}.  Giving up....")


def process_directory(pytorch2_file, sweeps_dir, args):
    directory = os.path.dirname(pytorch2_file)
    pytorch2_filename = os.path.basename(pytorch2_file)

    # Find all .py files in the same directory given we want to run these traces as well.
    py_files = [f for f in os.listdir(directory) if f.endswith(".py")]

    # Exclude the pytorch2.py file itself
    other_files = [f for f in py_files if f != pytorch2_filename]

    # Run tracing for pytorch2.py file
    module_name = get_module_name(pytorch2_file, sweeps_dir)
    run_commands(module_name, args)

    # Run tracing for all other .py files in the same directory
    for other_file in other_files:
        other_file_path = os.path.join(directory, other_file)
        other_module_name = get_module_name(other_file_path, sweeps_dir)
        run_commands(other_module_name, args)


def summarize_results(sqlite_databases):
    total_tests = 0
    total_passed = 0
    summary = []

    for db_file in sqlite_databases:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Check if the expected table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if not tables:
            conn.close()
            raise "Unable to find any tables in the the database"

        tables = sorted(tables)
        main_table = tables[0][0]

        basename_of_test = os.path.basename(db_file).replace(".sqlite", "")

        # Count total tests
        cursor.execute(f"SELECT COUNT(*) FROM {main_table}")
        total = cursor.fetchone()[0]
        total_tests += total

        # Count passed tests
        cursor.execute(f"SELECT COUNT(*) FROM {main_table} WHERE status='TestStatus.PASS'")
        passed = cursor.fetchone()[0]
        total_passed += passed

        if total > 0:
            pass_rate = 100 * passed / total
        else:
            pass_rate = 0

        summary.append([basename_of_test, total, passed, f"{pass_rate:.2f}"])

        conn.close()

    headers = ["Module", "Total Tests", "Total Passed", "Pass Rate (%)"]
    print(tabulate(summary, headers=headers, tablefmt="github"))

    if total_tests > 0:
        total_pass_rate = (total_passed / total_tests) * 100
    else:
        total_pass_rate = 0
    print(f"Total Tests: {total_tests}")
    print(f"Total Passed: {total_passed}")
    print(f"Pass Rate: {total_pass_rate:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )

    parser.add_argument(
        "--elastic",
        required=False,
        default="corp",
        help="Elastic Connection String for the vector and results database. Available presets are ['corp', 'cloud']",
    )

    parser.add_argument(
        "--suite-name", required=False, default="nightly", help="Suite of Test Vectors to run, or all tests if omitted."
    )

    home_dir = os.path.expanduser("~")
    default_sqlite_locations = Path(home_dir) / "eltwise"
    parser.add_argument(
        "--dump_path", required=False, default=f"{default_sqlite_locations}", help="Path to store the sqlite data"
    )

    folder_to_scan = Path(__file__).parent.parent.parent / "sweep_framework/sweeps/eltwise"

    parser.add_argument(
        "--folder",
        required=False,
        default=f"{folder_to_scan}",
        help="Folder to scan for any sweep tests ending in pytorch2.py",
    )
    args = parser.parse_args(sys.argv[1:])

    folder_to_scan = os.path.abspath(args.folder)

    # Find all pytorch2.py files
    pytorch2_files = find_pytorch2_files(folder_to_scan)

    sweeps_dir = os.path.abspath(Path(__file__).parent.parent.parent / "sweep_framework/sweeps")

    # Run commands for each pytorch2.py file and other .py files in the same directory
    for pytorch2_file in pytorch2_files:
        process_directory(pytorch2_file, sweeps_dir, args)

    sqlite_databases = [str(path) for path in Path(args.dump_path).rglob("*.sqlite")]

    summarize_results(sqlite_databases)
