import omegaconf

def load_config(config_path: str = "env_config.yaml"):
    return omegaconf.OmegaConf.load(config_path)
