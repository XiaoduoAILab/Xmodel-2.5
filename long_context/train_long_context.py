import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoModelForCausalLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from utils.data_utils import create_dataloaders


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
    def __post_init__(self):
        # 在父类__post_init__之前设置bf16_full_eval
        if self.bf16:
            self.bf16_full_eval = True
        super().__post_init__()


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, training_args):
        super(SupervisedDataset, self).__init__()

        data_config = [
            ("ultrafineweb-en", 0.18725),
            ("ultrafineweb-zh", 0.03716),
            ("starcoder", 0.02209),
            ("books", 0.00287),
            ("algebraic-stack", 0.01722),
            ("open-web-math", 0.01722),
            ("wiki", 0.01147),
            ("stackexchange", 0.00900),
            ("reddit", 0.00900),
            ("megawika", 0.01801),
            ("sft_mixed", 0.66870),
        ]

        rank = int(os.environ["LOCAL_RANK"])
        seed_offset = 42 + rank

        train_dataloader, val_dataloader = create_dataloaders(
            batch_size=1,
            block_size=training_args.model_max_length,
            data_config=data_config,
            sampling="random",
            train_data_dir=data_args.data_path,
            val_data_dir=None,
            seed=1337 + seed_offset,
            vocab_size=32000
        )
        self.dataloader_iter = iter(train_dataloader)

    def __len__(self):
        return 2 ** 32

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        batch = next(self.dataloader_iter)
        # print('batch.shape: ' + str(batch.shape))
        batch = batch[0]
        return dict(input_ids=batch, labels=batch)


def make_supervised_data_module(data_args, training_args):
    # 1. 先建完整 dataset
    full_dataset = SupervisedDataset(data_args=data_args, training_args=training_args)
    # 2. 让 Trainer 自己去 shard
    return {"train_dataset": full_dataset, "eval_dataset": None, "data_collator": None}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. 决定 dtype
    torch_dtype = (
        torch.float16 if training_args.fp16  else
        torch.bfloat16 if training_args.bf16 else
        torch.float32
    )

    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch_dtype, 
                                                 use_cache=False,
                                                 trust_remote_code=True)

    # 3. 数据 & Trainer
    data_module = make_supervised_data_module(data_args=data_args, training_args=training_args)
    trainer = Trainer(model=model, args=training_args, **data_module)

    # 4. 训练 & 保存模型
    trainer.train(resume_from_checkpoint=bool(list(Path(training_args.output_dir).glob("checkpoint-*"))))
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
