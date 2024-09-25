from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(config_path="configs", job_name="test", version_base="1.3")
cfg = compose(
    config_name="config", overrides=["experiment=test_many_train_single_val_test"]

)
print(OmegaConf.to_yaml(cfg, resolve=True))



