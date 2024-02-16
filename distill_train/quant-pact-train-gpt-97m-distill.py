from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample

from pathlib import Path
import wandb

from dataset_babylm import BabylmDataset

from models import GPT2LMHeadModel_pact


# Training Recipe
####################################################
NUM_EPOCHS = 1      # normal pretrain: 6
WARMUP_STEPS = 200  # normal pretrain: 200
LR = 1e-4        # normal pretrain: 2.5e-4
BATCH_SIZE = 32     # normal pretrain: 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0   # normal pretrain: 2.0
ALPHA = 0.5         # normal pretrain: 0.5
####################################################

# Quantization Recipe   # xuan: todo
####################################################
Weight_bit = "W4"
Activation_bit = "A4"
####################################################

# All Paths
########################################################################################################
student_dir = "/path/GPT-97M-distill-pretrain"
teacher_dir1 = "/path/Llama-360M"
teacher_dir2 = "/path/GPT2-705M"
tokenizer_path = "/path/gpt-clean-16000.json"
dataset_train_path = "/path/babylm_10M_clean"
dataset_eval_path = "/path/babylm_dev_clean"

MODEL_NAME = f'GPT-97M-pact-{Weight_bit}{Activation_bit}-epoch{NUM_EPOCHS}-warmup{WARMUP_STEPS}-lr{LR}-bs{BATCH_SIZE}-temp{TEMPERATURE}-alpha{ALPHA}'
MODEL_OUTPUT = Path('/path/checkpoints') / MODEL_NAME
########################################################################################################

EVAL_SAMPLES = 8192

tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

wandb_log = True

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(dataset_train_path, SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(dataset_eval_path, SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

# student = GPT2LMHeadModel_pact(config)
student = GPT2LMHeadModel_pact.from_pretrained(student_dir)


teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1)
teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)
teachers = [teacher1, teacher2]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # place each teacher on same device as student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # assert size
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
                loss_function(
                    F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                    F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
                )
                * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


if wandb_log:
    wandb.login()
    wandb.init(project='EdgeQAT', name=MODEL_NAME)

training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving
    report_to="wandb",
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=False,     # we do not have AMPERE GPU :(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
