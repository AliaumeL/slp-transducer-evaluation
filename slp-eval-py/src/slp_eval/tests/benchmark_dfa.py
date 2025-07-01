import time
import os
import matplotlib.pyplot as plt
import string
from slp_eval.compression_model import SLP
from slp_eval.lz78 import LZ78String
from slp_eval.automata import DFA
import gc
import random


def modk_a_dfa(k: int = 512) -> DFA[str, int]:
    """
    DFA that accepts strings where the number of 'a's is divisible by k.
    All other symbols are ignored (self-loop).
    """
    states: set[int] = set(range(k))
    delta: dict[tuple[int, str], int] = {}
    alphabet = list(string.ascii_lowercase)

    for i in range(k):
        for c in alphabet:
            if c == "a":
                delta[(i, "a")] = (i + 1) % k
            else:
                delta[(i, c)] = i  # self-loop on other chars
    
    start_state = 0
    final_states = {0}  # accept if mod k == 0
    return DFA(states, delta, final_states, start_state)

def dfa_accept_blue() -> DFA[str, int]:
    states = {0, 1, 2, 3, 4}  # 0=start, 4=accepting sticky
    start_state = 0
    final_states = {4}

    alphabet = set(string.ascii_uppercase + string.ascii_lowercase + string.whitespace + string.punctuation + string.digits)

    delta = {}

    # State 0: waiting for 'b' or 'B'
    for c in alphabet:
        if c.lower() == 'b':
            delta[(0, c)] = 1
        else:
            delta[(0, c)] = 0

    # State 1: waiting for 'l' or 'L'
    for c in alphabet:
        if c.lower() == 'l':
            delta[(1, c)] = 2
        elif c.lower() == 'b':
            delta[(1, c)] = 1  # allow repeated 'b's here
        else:
            delta[(1, c)] = 0

    # State 2: waiting for 'u' or 'U'
    for c in alphabet:
        if c.lower() == 'u':
            delta[(2, c)] = 3
        elif c.lower() == 'b':
            delta[(2, c)] = 1
        else:
            delta[(2, c)] = 0

    # State 3: waiting for 'e' or 'E'
    for c in alphabet:
        if c.lower() == 'e':
            delta[(3, c)] = 4  # accepting state
        elif c.lower() == 'b':
            delta[(3, c)] = 1
        else:
            delta[(3, c)] = 0

    # State 4: accepting state — stay here forever
    for c in alphabet:
        delta[(4, c)] = 4

    return DFA(states, delta, final_states, start_state)



def generate_slp_supercompress(n: int) -> SLP[str]:
    return SLP(constants=["a"], instructions=[(i, i) for i in range(n)])


def generate_lz_supercompress(n: int) -> LZ78String[str]:
    raw = ["a"] * (2 ** n)
    return LZ78String.from_list(raw)


def measure_time(func):
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return end - start


def benchmark_dfa_supercompress(dfa: DFA, max_exp: int):
    input_sizes = []
    dfa_on_slp_times = []
    dfa_on_lz_times = []
    decompress_slp_times = []
    decompress_lz_times = []
    lz_to_slp_times = []
    dfa_on_list_times = []

    for n in range(max_exp):
        print(f"Running for 2^{n} = {2 ** n} input length...")

        slp = generate_slp_supercompress(n)
        lz = generate_lz_supercompress(n)

        input_sizes.append(2 ** n)

        # DFA on SLP
        time_slp = measure_time(lambda: slp.run_dfa(dfa))
        dfa_on_slp_times.append(time_slp)

        # DFA on LZ78 → SLP
        time_lz = measure_time(lambda: lz.to_slp().run_dfa(dfa))
        dfa_on_lz_times.append(time_lz)

        # Decompress SLP then run DFA
        time_decompress_slp = measure_time(lambda: dfa.is_accepting(slp.evaluate()))
        decompress_slp_times.append(time_decompress_slp)

        # Decompress LZ78 then run DFA
        time_decompress_lz = measure_time(lambda: dfa.is_accepting(lz.to_list()))
        decompress_lz_times.append(time_decompress_lz)

        # LZ78 → SLP only
        time_lz_to_slp = measure_time(lambda: lz.to_slp())
        lz_to_slp_times.append(time_lz_to_slp)

        # DFA on plain list
        w = ["a"] * (2 ** n)
        time_list = measure_time(lambda: dfa.is_accepting(w))
        dfa_on_list_times.append(time_list)

    return input_sizes, dfa_on_slp_times, dfa_on_lz_times, decompress_slp_times, decompress_lz_times, lz_to_slp_times, dfa_on_list_times

def normalize_text(text: str) -> str:
    # Now remove all non-ASCII characters
    text = ''.join(ch for ch in text if 32 <= ord(ch) <= 126)
    return text

def benchmark_dfa_files(dfa: DFA, files_dir: str):
    file_sizes = []
    dfa_on_slp_times = []
    dfa_on_lz_times = []
    decompress_slp_times = []
    decompress_lz_times = []
    lz_to_slp_times = []
    dfa_on_list_times = []

    for filename in os.listdir(files_dir):
        file_path = os.path.join(files_dir, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            w = list(normalize_text(text))

        file_sizes.append(len(w))
        print(f"Running on file '{filename}' ({len(w)} symbols) ...")

        # Compress
        slp = SLP.list_to_slp(w)
        lz = LZ78String.from_list(w)
        gc.collect()


        # DFA on SLP
        time_slp = measure_time(lambda: slp.run_dfa(dfa))
        dfa_on_slp_times.append(time_slp)
        gc.collect()


        # DFA on LZ78 → SLP
        time_lz = measure_time(lambda: lz.to_slp().run_dfa(dfa))
        dfa_on_lz_times.append(time_lz)
        gc.collect()


        # Decompress SLP then run DFA
        time_decompress_slp = measure_time(lambda: dfa.is_accepting(slp.evaluate()))
        decompress_slp_times.append(time_decompress_slp)
        gc.collect()


        # Decompress LZ78 then run DFA
        time_decompress_lz = measure_time(lambda: dfa.is_accepting(lz.to_list()))
        decompress_lz_times.append(time_decompress_lz)
        gc.collect()


        # LZ78 → SLP only
        time_lz_to_slp = measure_time(lambda: lz.to_slp())
        lz_to_slp_times.append(time_lz_to_slp)
        gc.collect()


        # DFA on plain list
        time_list = measure_time(lambda: dfa.is_accepting(w))
        dfa_on_list_times.append(time_list)

        gc.collect()

    return file_sizes, dfa_on_slp_times, dfa_on_lz_times, decompress_slp_times, decompress_lz_times, lz_to_slp_times, dfa_on_list_times

import matplotlib.pyplot as plt

def plot_results(
    input_sizes,
    dfa_on_slp,
    dfa_on_lz,
    decompress_slp,
    decompress_lz,
    lz_to_slp,
    dfa_on_list,
    title="DFA Benchmark"
):
    # Zip everything together, sort by input size, then unzip again
    combined = list(zip(
        input_sizes,
        dfa_on_slp,
        dfa_on_lz,
        decompress_slp,
        decompress_lz,
        lz_to_slp,
        dfa_on_list
    ))
    combined.sort(key=lambda x: x[0])

    # Unpack the sorted results
    (
        input_sizes_sorted,
        dfa_on_slp,
        dfa_on_lz,
        decompress_slp,
        decompress_lz,
        lz_to_slp,
        dfa_on_list
    ) = zip(*combined)

    plt.figure(figsize=(10, 6))

    # Add markers for each line
    plt.plot(input_sizes_sorted, dfa_on_slp, label="DFA on SLP", marker="o")
    plt.plot(input_sizes_sorted, dfa_on_lz, label="DFA on LZ78", marker="s")
    plt.plot(input_sizes_sorted, decompress_slp, label="Decompress SLP + DFA", marker="^")
    plt.plot(input_sizes_sorted, decompress_lz, label="Decompress LZ78 + DFA", marker="v")
    plt.plot(input_sizes_sorted, lz_to_slp, label="LZ78 to SLP", marker="D")
    plt.plot(input_sizes_sorted, dfa_on_list, label="DFA on plain list", marker="x")

    plt.xlabel("Input size (number of symbols)")
    plt.ylabel("Execution time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("dfa_files_benchmark.png")
    plt.show()

import random

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


def benchmark_dfa_random_slp(dfa: DFA, max_limit: int, step_size: int):
    """
    Benchmark DFA on randomly generated SLPs of increasing size.
    max_limit: max number of instructions in SLP.
    step_size: instructions added per step.
    """
    results = []

    for num_instructions in range(step_size, max_limit + 1, step_size):
        slp = generate_random_compressible_slp(num_instructions)
        slp_size = 26 + num_instructions  # 26 base constants in your random generator
        input_size = len(slp.evaluate())
        lz = LZ78String.from_list(slp.evaluate())
        gc.collect()

        timings = {}

        # DFA on SLP directly
        start = time.perf_counter()
        slp.run_dfa(dfa)
        end = time.perf_counter()
        timings["dfa_on_slp"] = end - start

        # LZ78 compress the decompressed output
        

        # DFA on LZ78 converted to SLP
        start = time.perf_counter()
        lz.to_slp().run_dfa(dfa)
        end = time.perf_counter()
        timings["dfa_on_lz78"] = end - start

        # Decompress SLP then run DFA
        start = time.perf_counter()
        dfa.is_accepting(slp.evaluate())
        end = time.perf_counter()
        timings["decompress_slp"] = end - start

        # Decompress LZ78 then run DFA
        start = time.perf_counter()
        dfa.is_accepting(lz.to_list())
        end = time.perf_counter()
        timings["decompress_lz"] = end - start

        # LZ78 → SLP only
        start = time.perf_counter()
        lz.to_slp()
        end = time.perf_counter()
        timings["lz_to_slp"] = end - start

        # DFA on plain list
        w = slp.evaluate()
        start = time.perf_counter()
        dfa.is_accepting(w)
        end = time.perf_counter()
        timings["dfa_on_list"] = end - start

        results.append({
            "num_instructions": num_instructions,
            "slp_size": slp_size,
            "input_size": input_size,
            **timings
        })

        gc.collect()
        print(f"Benchmarked random SLP with DFA: {num_instructions} instructions, input length {input_size}, SLP size {slp_size}")

    return results


def plot_random_dfa(results):
    """
    Plot benchmark results from benchmark_random_dfa().
    Shows:
      1. Timing curves
      2. SLP size vs Input size
    """
    results = sorted(results, key=lambda r: r["slp_size"])
    slp_sizes = [r["slp_size"] for r in results]
    input_sizes = [r["input_size"] for r in results]
    dfa_on_slp = [r["dfa_on_slp"] for r in results]
    dfa_on_lz_slp = [r["dfa_on_lz78"] for r in results]
    decompress_slp = [r["decompress_slp"] for r in results]
    decompress_lz = [r["decompress_lz"] for r in results]
    lz_to_slp = [r["lz_to_slp"] for r in results]
    dfa_on_list = [r["dfa_on_list"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # --- Timing plot ---
    ax1.plot(slp_sizes, dfa_on_slp, label="DFA on SLP")
    ax1.plot(slp_sizes, dfa_on_lz_slp, label="DFA on LZ78 → SLP")
    ax1.plot(slp_sizes, decompress_slp, label="Decompress SLP + DFA")
    ax1.plot(slp_sizes, decompress_lz, label="Decompress LZ78 + DFA")
    ax1.plot(slp_sizes, lz_to_slp, label="LZ78 → SLP only")
    ax1.plot(slp_sizes, dfa_on_list, label="DFA on list")

    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_title("DFA Benchmark: Random SLPs - Timing")
    ax1.legend()
    ax1.grid(True)

    # --- SLP size vs Input size ---
    ax2.plot(slp_sizes, input_sizes, marker='o')
    ax2.set_xlabel("SLP size")
    ax2.set_ylabel("Input size (length of expansion)")
    ax2.set_title("SLP Size vs Input Size")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("dfa_random_benchmark.png")
    plt.show()




if __name__ == "__main__":

    # dfa = modk_a_dfa(512)
    # max_exp = 25

    # (
    #     input_sizes,
    #     dfa_on_slp,
    #     dfa_on_lz,
    #     decompress_slp,
    #     decompress_lz,
    #     lz_to_slp,
    #     dfa_on_list
    # ) = benchmark_dfa_supercompress(dfa, max_exp)

    # plot_results(input_sizes, dfa_on_slp, dfa_on_lz, decompress_slp, decompress_lz, lz_to_slp, dfa_on_list, title="DFA Supercompress Benchmark")

    # Benchmark files too:
    # dfa = dfa_accept_blue()
    # files_dir = "./slp_eval/tests/files"
    # (
    #     file_sizes,
    #     f_dfa_on_slp,
    #     f_dfa_on_lz,
    #     f_decompress_slp,
    #     f_decompress_lz,
    #     f_lz_to_slp,
    #     f_dfa_on_list
    # ) = benchmark_dfa_files(dfa, files_dir)

    # plot_results(file_sizes, f_dfa_on_slp, f_dfa_on_lz, f_decompress_slp, f_decompress_lz, f_lz_to_slp, f_dfa_on_list, title="DFA Files Benchmark")

    dfa = modk_a_dfa(20)
    max_limit = 270
    step_size = 10
    results = benchmark_dfa_random_slp(dfa, max_limit, step_size)
    plot_random_dfa(results)