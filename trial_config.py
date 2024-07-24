import configparser

def read_config(config_file):
    """
    Reads the configuration from the given INI file and returns a dictionary
    of configurations.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    settings = {
        'name': config.get('Trial', 'name'),

        'directory': config.get('Paths', 'directory'),
        'corpora_dir': config.get('Paths', 'corpora_dir'),
        'cleaned_dir': config.get('Paths', 'cleaned_dir'),
        'fuzz_dir': config.get('Paths', 'fuzz_dir'),
        'score_dir': config.get('Paths', 'score_dir'),

        'max_errors': config.getint('Settings', 'max_errors'),
        'n_simulations': config.getint('Settings', 'n_simulations'),
        'ks_a': config.getfloat('Settings', 'ks_a'),
        'k': config.getint('Settings', 'k'),
        'limit_atheris': config.getboolean('Settings', 'limit_atheris'),
        'atheris_entries': config.getint('Settings', 'atheris_entries'),
        'batch_size': config.getint('Settings', 'batch_size'),

        'fuzzer': config.get('Fuzzing', 'fuzzer'),
        'max_sample_size': config.getint('Fuzzing', 'max_sample_size'),
        'number_of_samples': config.getint('Fuzzing', 'number_of_samples'),
        'number_of_variants': config.getint('Fuzzing', 'number_of_variants'),
        'number_of_steps': config.getint('Fuzzing', 'number_of_steps'),

        'alpha_embed': config.getfloat('Score Coefficients', 'alpha_embed'),
        'beta_scaled': config.getfloat('Score Coefficients', 'beta_scaled'),
        'gamma_anom': config.getfloat('Score Coefficients', 'gamma_anom'),

        'scaling_factors': {
            'Output_Size_B': config.getfloat('Scaling', 'Output_Size_B'),
            'Output_Entropy_bits_per_symbol': config.getfloat('Scaling', 'Output_Entropy_bits_per_symbol'),
            'Top_Function_Calls': config.getfloat('Scaling', 'Top_Function_Calls'),
            'Peak_Memory_B': config.getfloat('Scaling', 'Peak_Memory_B'),
            'Final_Memory_B': config.getfloat('Scaling', 'Final_Memory_B'),
            'CPU_ms': config.getfloat('Scaling', 'CPU_ms'),
            'SLOC': config.getfloat('Scaling', 'SLOC'),
            'Instructions': config.getfloat('Scaling', 'Instructions'),
        }
    }

    assert settings["alpha_embed"] >= 0, "alpha must be >= 0, check {config_file}"
    assert settings["beta_scaled"] >= 0, "beta must be >= 0, check {config_file}"
    assert settings["gamma_anom"] >= 0, "gamma must be >= 0, check {config_file}"

    return settings

