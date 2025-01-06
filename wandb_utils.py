import os
import wandb
from wandb_key import WANDB_KEY

os.environ["WANDB_API_KEY"] = WANDB_KEY

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=config["name"],
    )
    wandb.run.save()
