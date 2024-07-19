import os
import random
import argparse
from trial_config import read_config

def random_mutation(sample, max_size):
    operations = ['insert', 'delete', 'modify']
    operation = random.choice(operations)
    
    if operation == 'insert' and len(sample) < max_size:
        pos = random.randint(0, len(sample))
        char = bytes([random.randint(0, 255)])
        sample = sample[:pos] + char + sample[pos:]
    
    elif operation == 'delete' and len(sample) > 1:
        pos = random.randint(0, len(sample) - 1)
        sample = sample[:pos] + sample[pos+1:]
    
    elif operation == 'modify' and len(sample) > 0:
        pos = random.randint(0, len(sample) - 1)
        char = bytes([random.randint(0, 255)])
        sample = sample[:pos] + char + sample[pos+1:]
    
    return sample

def process_files(input_dir, output_dir, max_size, num_mutations, num_samples):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        with open(input_file, 'rb') as f:
            sample = f.read()
        
        for i in range(num_samples):
            mutated_sample = sample
            for j in range(num_mutations):
                mutated_sample = random_mutation(mutated_sample, max_size)
                output_file = os.path.join(output_dir, f"mutated_{i+1}_{j+1}_{filename}")
                with open(output_file, 'wb') as f:
                    f.write(mutated_sample)

def main():
    parser = argparse.ArgumentParser(description='Mutate samples.')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to the configuration file (default: config.ini)')
    parser.add_argument('-s', '--split', action='store_true', 
                        help='Split the files to accomodate threading')
    parser.add_argument('input_dir', type=str, help='Location of the samples to mutate')
    parser.add_argument('output_dir', type=str, help='Directory to output the files')
    args = parser.parse_args()

    config = read_config(args.config)
    print(args.split)
    print(args.input_dir)
    print(args.output_dir)
    print(config["max_sample_size"], config["number_of_steps"], config["number_of_variants"])
    process_files(args.input_dir, args.output_dir, config["max_sample_size"], config["number_of_steps"], config["number_of_variants"])

if __name__ == "__main__":
    main()

