import argparse
import importlib

import cProfile
import pstats
# import psutil
import tracemalloc
import trace

import re
import os
import io
import sys
import time
import gc
import tempfile
import uuid

from IPython.display import display
import numpy as np
import pandas as pd

from status import Status
from encoding import calculate_entropy

# Get the instruction count from cprofile
def cprofile_output(func_name, file_loc, profiling_stats_filename):
    val = 0
    output = io.StringIO()
    profile_output = ""
    error_output = ""
    
    try:
        cProfile.run(f"{func_name}('{file_loc}')", profiling_stats_filename)
        #val = mystdout.getvalue() # getNumCalls(mystdout.getvalue())
    except Exception as ex:
        error_output = str(ex)
    p = pstats.Stats(profiling_stats_filename, stream=output)
    p.sort_stats('calls')
    p.print_stats()
    output.seek(0)
    profile_output = output.read()
    return profile_output, error_output

def trace_function(func, file_loc):
    # Create a Trace object, telling it what to trace
    tracer = trace.Trace(count=False, trace=True)  # count=False to not count the number of executions

    # Use a temporary file to store the trace output
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        # Redirect stdout to suppress the trace output
        with open(temp_filename, 'w') as temp_file:
            original_stdout = sys.stdout
            sys.stdout = temp_file
            try:
                # Run the function with tracing enabled
                tracer.runfunc(func, file_loc)
            except Exception as e:
                print(f"Exception occurred during function execution: {e}")
            finally:
                sys.stdout = original_stdout

        with open(temp_filename, 'r') as f:
            trace_output = f.read()

        trace_lines = trace_output.splitlines()
        len_all = len(trace_lines)
        executed_lines = set(trace_lines)

        return len(executed_lines), len_all
    finally:
        os.remove(temp_filename)

# Get the output from the function as a display object
# This can often have richer information
def capture_display_output(obj):
    # Create a string buffer to capture the output
    buffer = io.StringIO()
    
    # Redirect the standard output to the buffer
    sys.stdout = buffer
    
    try:
        # Use the display function to output the object
        display(obj)
    finally:
        # Reset the standard output to the original value
        sys.stdout = sys.__stdout__
    
    # Get the captured output as a string
    output = buffer.getvalue()
    
    # Close the buffer
    buffer.close()
    
    return output

# Since most things run too quick we utilize repetitions
def measure_cpu_usage(func, file_loc, num_runs=1, repetitions=10):
    cpu_usages = []
    for _ in range(num_runs):
        # process = psutil.Process()

        # Capture CPU times before running the function
        start_time = time.process_time()
        
        for _ in range(repetitions):
            try:
                func(file_loc)
            except Exception as e:
                pass

        # Capture CPU times after running the function
        end_time = time.process_time()

        # Calculate CPU usage for the process
        cpu_usage = (end_time - start_time) * 1000

        cpu_usages.append(cpu_usage)

    return np.mean(cpu_usages)

# Use tracemalloc to get memory peak
def measure_peak_memory(func, file_loc):
    tracemalloc.start()
    try:
        func(file_loc)
    except Exception as e:
        pass
        #print(f"Exception occurred: {e}")
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, current

# Run all tests on samples in a given directory
def run_samples(func, func_name, directory, name, profiling_stats_filename, repetitions=10, log_every=1000):
 #   Status.write(f"\n\nTesting function {func_name}\n")
    acc_duration = 0
    # Create a quick baseline to initialize cProfile
    cprofile_output(func_name, "/dev/null", profiling_stats_filename)
    results = []
    
    dir_list = os.listdir(directory)
    dir_len = len(dir_list)
    for i, filename in enumerate(dir_list):
        file_location = os.path.join(directory, filename)
        start_time = time.time()
        result, error_msg = cprofile_output(func_name, file_location, profiling_stats_filename)

        sloc, all_lines = trace_function(func, file_location)
        average_cpu_usage = measure_cpu_usage(func, file_location, repetitions=repetitions)
        peak_mem, end_mem = measure_peak_memory(func, file_location)
        output = ""
        try: 
            output = capture_display_output(func(file_location))
        except Exception:
            pass
        
        end_time = time.time()

        duration = end_time - start_time
        acc_duration += duration / 60 # convert to mins
        duration = duration * 1000
        
        function_calls = 0
        top_function_num = 0
        top_function_name = ''
        next_line_is_top = False
        for l in result.split("\n"):
            # Extract information about the function that was called the most
            if next_line_is_top:
                l = l.strip()
                l = re.sub(r'\s+', ' ', l)
                ls = l.split(" ")
                top_function_num = ls[0]
                top_function_name = ls[-1]
                break
            
            if "function calls" in l: function_calls = int(l.strip().split(" ")[0])
            elif "ncalls" in l: next_line_is_top = True 

        results.append( [ func_name, directory, filename
                        , error_msg, output, len(output), calculate_entropy(output)
                        , function_calls, top_function_name, top_function_num
                        , peak_mem, end_mem
                        , average_cpu_usage, repetitions
                        , sloc, all_lines
                        , duration ])
        
        if i % log_every == 0: 
            Status.write(f"{name}: {i}/{dir_len} ({100*i/dir_len:.2f}%) -- {acc_duration:0.2f}m\n")
            acc_duration = 0
    
    return results

# Load the function under test
def load_function(args):
    sys.path.append(os.path.abspath(args.harness_dir))
    module = importlib.import_module(args.filename)
    func = getattr(module, args.func_name)
    globals()[args.func_name] = func
    return func

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run a sample function with given parameters to get resource values.")
    parser.add_argument('harness_dir', type=str, help='Location of the harness.')
    parser.add_argument('filename', type=str, help='The name of the file to process.')
    parser.add_argument('func_name', type=str, help='The name of the function to run.')
    parser.add_argument('mod_name', type=str, help='The name of the module the function is in.')
    parser.add_argument('directory', type=str, help='The directory where the input samples are located.')
    
    parser.add_argument('--out_file', type=str, default="resources", help='Location to save results dataframe.')
    parser.add_argument('--log_every', type=int, default=1000, help='How often to log info (default: 1000 samples)')
    parser.add_argument('--repetitions', type=int, default=10, help='Times to run for CPU accumulation (default: 10)')
    
    args = parser.parse_args()
    func = load_function(args)

    # Call the function with parsed arguments
    rows = [ "Function", "Corpus", "Sample"
           , "Error", "Output", "Output Size (B)", "Output Entropy (bits per symbol)"
           , "Function Calls", "Top Function", "Top Function Calls" 
           , "Peak Memory (B)", "Final Memory (B)"
           , "CPU (ms)", "Repetitions"
           , "SLOC", "Instructions" 
           , "Duration (ms)"
           ]
 
    # Create a unique filename to avoid multi-process conflicts
    profiling_stats_filename = f"profiling_stats/{uuid.uuid4().hex}.prof"

    results = run_samples(func, args.func_name, args.directory, args.filename, profiling_stats_filename, args.repetitions, args.log_every)
    os.remove(profiling_stats_filename)

    df = pd.DataFrame(results, columns=rows)
    def replace_surrogates(s):
        return ''.join([char if not 0xD800 <= ord(char) <= 0xDFFF else '?' for char in s])
    df['Output'] = df['Output'].apply(replace_surrogates)
    df.to_parquet(f"{args.out_file}.parquet", index=False)
    html = f"<style>body {{ color: #ccc; }}</style>\n{df.to_html()}"

    with open(f"{args.out_file}.html", 'w') as file: file.write(html)
    Status.write("::Completed::")
    
if __name__ == "__main__":
    main()
