import subprocess
import socket
import re
import pandas as pd
import os


def get_lab_machine_name():
    # Get the hostname (Docker container name)
    hostname = socket.gethostname()
    # Extract the part before '-special'
    match = re.match(r"^(.*?)\-special", hostname)
    return match.group(1) if match else hostname


def run_first_script():
    command = 'PYTHONPATH=tests/sweep_framework python -m tracy -r -m -p "pytest tests/sweep_framework/perf_model/test_relu_perf.py"'
    subprocess.run(command, shell=True, check=True)


def run_second_script(lab_machine_name):
    command = f"python tests/sweep_framework/perf_model/make_common_stats.py {lab_machine_name}"
    subprocess.run(command, shell=True, check=True)


def run_third_script():
    command = "python tests/sweep_framework/perf_model/make_all_machine_stats.py"
    subprocess.run(command, shell=True, check=True)


def sort_all_machines_results():
    output_csv_file = "all_machines_results.csv"

    if os.path.isfile(output_csv_file):
        # Load existing results
        df = pd.read_csv(output_csv_file)
        # Sort by the 'first' value
        df_sorted = df.sort_values(by="first")
        # Save the sorted results back to the output file
        df_sorted.to_csv(output_csv_file, index=False)
        print(f"Sorted data in {output_csv_file}")


if __name__ == "__main__":
    try:
        run_first_script()

        lab_machine_name = get_lab_machine_name()
        run_second_script(lab_machine_name)

        run_third_script()

        # Sort the results after running the scripts
        sort_all_machines_results()

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing a script: {e}")
