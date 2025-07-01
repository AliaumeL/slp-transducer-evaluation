from slp_eval.transducer_models import SST
from slp_eval.compression_model import SLP
import string
import time
import matplotlib.pyplot as plt
import os
import gc
import random

def replace_every_2nd_a_sst() -> SST[str, str, str, str]:
    """
    SST that replaces every 2nd 'a' with 'b'.
    Example: 'aaaa' -> 'abab'
    """
    alphabet = set(string.ascii_lowercase)

    states = {"q0", "q1"}
    registers = {"R"}
    init_state = "q0"
    init_regs = {"R": []}

    delta = {}
    reg_update = {}

    for c in alphabet:
        if c == "a":
            # State q0: even number of 'a's seen so far, keep 'a', go to q1
            delta[("q0", "a")] = "q1"
            reg_update[("q0", "a")] = {"R": ["R", "a"]}

            # State q1: odd number of 'a's seen so far, replace with 'b', go back to q0
            delta[("q1", "a")] = "q0"
            reg_update[("q1", "a")] = {"R": ["R", "b"]}
        else:
            # Any other char: stay in same state, append as-is
            delta[("q0", c)] = "q0"
            reg_update[("q0", c)] = {"R": ["R", c]}

            delta[("q1", c)] = "q1"
            reg_update[("q1", c)] = {"R": ["R", c]}

    output_fn = {
        "q0": ["R"],
        "q1": ["R"],
    }

    sst = SST(
        states=states,
        registers=registers,
        input_lang=alphabet,
        output_lang=alphabet,
        delta=delta,
        reg_update=reg_update,
        output_fn=output_fn,
        init_state=init_state,
        init_regs=init_regs
    )

    return sst

def words_after_blue_sst() -> SST[str, str, str, int]:

    alphabet = set(string.ascii_uppercase + string.ascii_lowercase + string.whitespace + string.punctuation + string.digits)
    
    states = {
        "q0", "b1", "b2", "b3", "b4",        # Detecting "blue"
        "skip_space", "cap", "done"         # Capturing next word
    }

    registers = {1}
    init_state = "q0"
    init_regs = {1: []}

    delta = {}
    reg_update = {}

    # q0: waiting for 'b'
    for c in alphabet:
        delta[("q0", c)] = "b1" if (c == "b" or c == "B") else "q0"
        reg_update[("q0", c)] = {1: [1]}

    # b1: waiting for 'l'
    for c in alphabet:
        if c == "l" or c == "L":
            delta[("b1", c)] = "b2"
        elif c == "b" or c == "B":
            delta[("b1", c)] = "b1"
        else:
            delta[("b1", c)] = "q0"

        reg_update[("b1", c)] = {1: [1]}

    # b2: waiting for 'u'
    for c in alphabet:
        if c == "u" or c == "U":
            delta[("b2", c)] = "b3"
        elif c == "b" or c == "B":
            delta[("b2", c)] = "b1"
        else:
            delta[("b2", c)] = "q0"

        reg_update[("b2", c)] = {1: [1]}

    # b3: waiting for 'e'
    for c in alphabet:
        if c == "e" or c == "E":
            delta[("b3", c)] = "b4"
        elif c == "b" or c == "B":
            delta[("b3", c)] = "b1"
        else:
            delta[("b3", c)] = "q0"

        reg_update[("b3", c)] = {1: [1]}

    # b4: waiting for space after "blue"
    for c in alphabet:
        if c in string.punctuation or c in string.whitespace:
            delta[("b4", c)] = "skip_space"
        elif c == "b" or c == "B":
            delta[("b4", c)] = "b1"
        else:
            delta[("b4", c)] = "q0"

        reg_update[("b4", c)] = {1: [1]}

    # skip_space: skip spaces after "blue"
    for c in alphabet:
        if c in string.punctuation or c in string.whitespace:
            delta[("skip_space", c)] = "skip_space"
            reg_update[("skip_space", c)] = {1: [1]}
        else:
            delta[("skip_space", c)] = "cap"
            reg_update[("skip_space", c)] = {1: [1, c]}  # start capturing

    # cap: capture next word (until space)
    for c in alphabet:
        if c in string.punctuation or c in string.whitespace:
            delta[("cap", c)] = "done"
            reg_update[("cap", c)] = {1: [1]}
        else:
            delta[("cap", c)] = "cap"
            reg_update[("cap", c)] = {1: [1, c]}

    # done: output register r, then reset
    for c in alphabet:
        if c == "b" or c == "B":
            delta[("done", c)] = "b1"
        else:
            delta[("done", c)] = "q0"
            
        reg_update[("done", c)] = {1: [1, " "]}

    # Output function: emit register r only in done state
    output_fn = {
        "q0": [1],
        "b1": [1],
        "b2": [1],
        "b3": [1],
        "b4": [1],
        "skip_space": [1],
        "cap": [1],
        "done": [1]
    }

    # Instantiate the SST
    sst = SST(
        states=states,
        registers=registers,
        input_lang=alphabet,
        output_lang=alphabet,
        init_state=init_state,
        init_regs=init_regs,
        delta=delta,
        reg_update=reg_update,
        output_fn=output_fn #type: ignore
    )

    return sst

def generate_slp_supercompress(n: int) -> SLP[str]:
    """n represents number in a^{2^n}"""
    slp = SLP(
        constants=["a"],
        instructions=[(i,i) for i in range(n)],
    ) 
    return slp

def benchmark_supercompress(sst: SST, max_exp: int):
    """
    Benchmarks SST on inputs of form a^{2^n}, compressed as an SLP.
    Returns a list of dicts:
    {
        "exp": int,
        "input_size": int,
        "decompress_and_run": float,
        "sst_dumb": float,
        "sst_on_slp": float,
        "output_compression": float
    }
    """
    results = []

    for n in range(max_exp):
        slp = generate_slp_supercompress(n)
        input_size = 2 ** n

        timings = {}

        # # Decompress and run
        # start = time.perf_counter()
        # slp.evaluate_by_uncompress(sst)
        # end = time.perf_counter()
        # timings["decompress_and_run"] = end - start

        # # SST dumb
        # start = time.perf_counter()
        # slp.run_sst_dumb(sst)
        # end = time.perf_counter()
        # timings["sst_dumb"] = end - start

        # SST on SLP
        start = time.perf_counter()
        slp.run_sst_on_slp(sst)
        end = time.perf_counter()
        timings["sst_on_slp"] = end - start

        # SST output compression
        start = time.perf_counter()
        slp.run_sst_output_compression(sst, "")
        end = time.perf_counter()
        timings["output_compression"] = end - start

        results.append({
            "exp": n,
            "input_size": input_size,
            **timings
        })

        print(f"Benchmarked 2^{n}")

    return results

def plot_supercompress(results):
    """
    Plots results from benchmark_supercompress.
    """
    # Sort results by input_size ascending
    results = sorted(results, key=lambda r: r["input_size"])

    input_sizes = [r["input_size"] for r in results]
    # decompress_and_run = [r["decompress_and_run"] for r in results]
    # sst_dumb = [r["sst_dumb"] for r in results]
    sst_on_slp = [r["sst_on_slp"] for r in results]
    output_compression = [r["output_compression"] for r in results]

    plt.figure(figsize=(12, 7))
    # plt.plot(input_sizes, decompress_and_run, label="Decompress and run")
    # plt.plot(input_sizes, sst_dumb, label="SST dumb")
    plt.plot(input_sizes, sst_on_slp, label="SLP-compressed SST run")
    plt.plot(input_sizes, output_compression, label="SLP-compressed SST run with output compression")

    plt.xlabel("Input Size (Number of characters)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("SST Benchmark: Super-compression")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("sst_supercompress_benchmark.png")
    plt.show()

def normalize_text(text: str) -> str:
    # Now remove all non-ASCII characters
    text = ''.join(ch for ch in text if 32 <= ord(ch) <= 126)
    return text

def benchmark_files(sst: SST, files_dir):
    results = []

    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        normalized_text = normalize_text(text)
        slp = SLP.list_to_slp(list(normalized_text))

        input_size = len(normalized_text)
        timings = {}

        # Decompress and run
        start = time.perf_counter()
        sst.run_on_list(list(normalized_text))
        end = time.perf_counter()
        timings["run_on_list"] = end - start
        # SST dumb
        start = time.perf_counter()
        slp.run_sst_dumb(sst)
        end = time.perf_counter()
        timings["sst_dumb"] = end - start
        # SST on SLP
        start = time.perf_counter()
        slp.run_sst_on_slp(sst)
        end = time.perf_counter()
        timings["sst_on_slp"] = end - start
        # SST output compression
        start = time.perf_counter()
        slp.run_sst_output_compression(sst, "")
        end = time.perf_counter()
        timings["output_compression"] = end - start

        results.append({
            "file": filename,
            "input_size": input_size,
            **timings
        })

        print(f"Benchmarked {filename}")
        gc.collect()

    return results    

def plot_file_benchmarks(results):
    """
    Plots the benchmark results from benchmark_files().
    Expects results to be a list of dicts:
      {
          "file": filename,
          "input_size": int,
          "run_on_list": float,
          "sst_dumb": float,
          "sst_on_slp": float,
          "output_compression": float
      }
    """
    results = sorted(results, key=lambda r: r["input_size"])
    input_sizes = [r["input_size"] for r in results]
    decompress_and_run = [r["run_on_list"] for r in results]
    sst_dumb = [r["sst_dumb"] for r in results]
    sst_on_slp = [r["sst_on_slp"] for r in results]
    output_compression = [r["output_compression"] for r in results]
    filenames = [r["file"] for r in results]

    plt.figure(figsize=(12, 7))

    plt.plot(input_sizes, decompress_and_run, 'o-', label="Run on list")
    plt.plot(input_sizes, sst_dumb, 's-', label="SST dumb")
    plt.plot(input_sizes, sst_on_slp, '^-', label="SLP-compressed SST run")
    plt.plot(input_sizes, output_compression, 'x-', label="SLP-compressed SST run with output compression")

    for i, name in enumerate(filenames):
        plt.annotate(name, (input_sizes[i], decompress_and_run[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Input Size (number of characters)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("SST Benchmark: Real Files")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sst_files_benchmark.png")
    plt.show()

def generate_random_compressible_slp(n: int) -> SLP[str]:
    """
    Generates an SLP that expands to ~2^n, uses multiple constants,
    and enforces deeper structure: left/right are always chosen
    from instructions only if possible.
    """
    constants = list("abcdefghijklmnopqrstuvwxyz")
    instructions = []
    num_constants = 26
    current_size = num_constants

    for i in range(n):
        threshold = current_size - num_constants
        left = random.randint(threshold, current_size - 1)
        right = random.randint(threshold, current_size - 1)

        instructions.append((left, right))
        current_size += 1

    return SLP(constants=constants, instructions=instructions)

def benchmark_random_slps(sst: SST, max_limit: int, step_size: int):
    """
    Benchmark SST on randomly generated SLPs of increasing size.
    max_steps: how many instructions at max.
    step_size: instructions added per step.
    """
    results = []

    for num_instructions in range(step_size, max_limit + 1, step_size):
        slp = generate_random_compressible_slp(num_instructions)
        input_size = len(slp.evaluate())
        slp_size = 26 + num_instructions

        timings = {}

        start = time.perf_counter()
        slp.evaluate_by_uncompress(sst)
        end = time.perf_counter()
        timings["decompress_and_run"] = end - start

        start = time.perf_counter()
        slp.run_sst_dumb(sst)
        end = time.perf_counter()
        timings["sst_dumb"] = end - start

        start = time.perf_counter()
        slp.run_sst_on_slp(sst)
        end = time.perf_counter()
        timings["sst_on_slp"] = end - start

        start = time.perf_counter()
        slp.run_sst_output_compression(sst, "")
        end = time.perf_counter()
        timings["output_compression"] = end - start

        results.append({
            "num_instructions": num_instructions,
            "slp_size": slp_size,
            "input_size": input_size,
            **timings
        })

        gc.collect()

        print(f"Benchmarked random SLP: {num_instructions} instructions, input length {input_size}, slp size {slp_size}")

    return results

import matplotlib.pyplot as plt

def plot_random_slps(results):
    """
    Plot benchmark results from benchmark_random_slps().
    Shows:
      1. Timing curves
      2. SLP size vs Input size
    """
    results = sorted(results, key=lambda r: r["slp_size"])
    slp_sizes = [r["slp_size"] for r in results]
    input_sizes = [r["input_size"] for r in results]
    decompress_and_run = [r["decompress_and_run"] for r in results]
    sst_dumb = [r["sst_dumb"] for r in results]
    sst_on_slp = [r["sst_on_slp"] for r in results]
    output_compression = [r["output_compression"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Timing plot ---
    ax1.plot(slp_sizes, decompress_and_run, label="Decompress and run")
    ax1.plot(slp_sizes, sst_dumb, label="SST dumb")
    ax1.plot(slp_sizes, sst_on_slp, label="SLP-compressed SST run")
    ax1.plot(slp_sizes, output_compression, label="SLP-compressed SST run with output compression")

    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("SST Benchmark: Random SLPs - Timing")
    ax1.legend()
    ax1.grid(True)

    # --- SLP size vs Input size ---
    ax2.plot(slp_sizes, input_sizes, marker='o')
    ax2.set_xlabel("SLP size")
    ax2.set_ylabel("Input size (length of expansion)")
    ax2.set_title("SLP Size vs Input Size")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("sst_random_benchmark.png")
    plt.show()



if __name__ == "__main__":

    sst = replace_every_2nd_a_sst()
    max_exp = 24
    print("Benchmark Supercompress")
    results = benchmark_supercompress(sst, max_exp)
    plot_supercompress(results)

    # sst = words_after_blue_sst()
    # print("Benchmark Text Files")
    # results = benchmark_files(sst, "./slp_eval/tests/files")
    # plot_file_benchmarks(results)

    # sst = replace_every_2nd_a_sst()
    # print("Benchmark Random SLPs")
    # results = benchmark_random_slps(sst, max_limit=210, step_size=10)
    # plot_random_slps(results)