#!/usr/bin/env python3
"""
Qwen2.5-Coder-7B QLoRA Fine-tuning for RTL Generation

Fine-tunes Qwen2.5-Coder-7B-Instruct on RTL dataset using QLoRA.
Qwen2.5-Coder is specifically optimized for code generation including HDL.

QLoRA Parameters:
- 4-bit quantization (NF4)
- LoRA rank: 64
- LoRA alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset


CATEGORY_STAGE_MAP = {
    1: {"simple_module", "arithmetic"},
    2: {"simple_module", "arithmetic", "fsm", "memory"},
    3: None  # Stage 3 uses entire dataset
}

LABEL_HIERARCHY = {
    "Sequential": {
        "FSM": ["Mealy", "Moore", "SequenceDetector", "Handshake", "Other"],
        "Controller": ["Pipeline", "Arbiter", "DMA", "CacheCtrl", "BusCtrl", "Other"],
        "Counter": ["Up", "Down", "UpDown", "WithLoad", "WithEnable", "Saturating", "Other"],
        "ShiftRegister": ["SISO", "SIPO", "PISO", "PIPO", "Bidirectional", "Universal"],
        "Memory": ["RegisterFile", "FIFO_Sync", "FIFO_Async", "LIFO", "RAM_SinglePort", "RAM_DualPort", "ROM", "CAM"]
    },
    "Combinational": {
        "Arithmetic": ["Adder", "Subtractor", "Multiplier", "Divider", "ALU", "Comparator", "Bitwise"],
        "MuxDemux": ["Multiplexer", "Demultiplexer"],
        "Codecs": ["Decoder", "Encoder", "PriorityEncoder"],
        "Logic": ["Gates", "Parity", "Other"],
        "Glue": ["Adapter", "Wrapper", "Bridge"]
    },
    "Interface": {
        "Protocol": ["AXI4", "AXI4_Lite", "AXI_Stream", "APB", "UART_Tx", "UART_Rx", "SPI_Master", "SPI_Slave", "I2C_Controller", "I2C_Target", "Wishbone"]
    },
    "TimingClocking": {
        "Clocking": ["ClockDivider", "BaudGen", "PLLStub", "ClockGater"],
        "Reset": ["SyncReset", "AsyncReset"]
    },
    "CDC": {
        "Synchronizers": ["TwoFFSync", "PulseSync", "AsyncFIFO"]
    },
    "EdgeDetect": {
        "Edge": ["RiseDetect", "FallDetect", "BothEdge"]
    },
    "Verification": {
        "Testbench": ["SimulationSV", "Cocotb"],
        "Formal": ["SBY_ResetSafety", "SBY_OneHot", "SBY_CustomSet"]
    }
}

L1_LABELS = list(LABEL_HIERARCHY.keys())


def _norm(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return ''.join(label.replace('-', ' ').replace('_', ' ').split()).lower()


def _build_l2_labels() -> List[str]:
    labels = []
    for l1, l2_dict in LABEL_HIERARCHY.items():
        for l2 in l2_dict.keys():
            labels.append(f"{l1}/{l2}")
    return labels


def _build_l3_labels() -> List[str]:
    labels = []
    for l1, l2_dict in LABEL_HIERARCHY.items():
        for l2, l3_list in l2_dict.items():
            for l3 in l3_list:
                labels.append(f"{l1}/{l2}/{l3}")
    return labels


L2_LABELS = _build_l2_labels()
L3_LABELS = _build_l3_labels()

L1_CANONICAL = {_norm(label): label for label in L1_LABELS}
L2_CANONICAL = {_norm(label.split('/', 1)[1]): label for label in L2_LABELS}
# For L3 we need combination of canonical parent/child; build nested map
L3_CANONICAL: Dict[str, str] = {}
for label in L3_LABELS:
    _, _, leaf = label.split('/', 2)
    key = _norm(leaf)
    L3_CANONICAL.setdefault(key, label)

L1_TO_ID = {label: idx for idx, label in enumerate(L1_LABELS)}
L2_TO_ID = {label: idx for idx, label in enumerate(L2_LABELS)}
L3_TO_ID = {label: idx for idx, label in enumerate(L3_LABELS)}

TB_TYPES = ["template", "a7_llm", "cocotb", "none"]
TB_TYPE_TO_ID = {name: idx for idx, name in enumerate(TB_TYPES)}
TB_TYPE_CANONICAL = {_norm(name): name for name in TB_TYPES}
TB_TYPE_CANONICAL.update({
    "llmqwen": "a7_llm",
    "llm": "a7_llm"
})


def canonicalize_l1(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return L1_CANONICAL.get(_norm(value))


def canonicalize_l2(l1: Optional[str], value: Optional[str]) -> Optional[str]:
    if not l1 or not value:
        return None
    normalized = _norm(value)
    for candidate in LABEL_HIERARCHY.get(l1, {}):
        if _norm(candidate) == normalized:
            return f"{l1}/{candidate}"
    return None


def canonicalize_l3(l1: Optional[str], l2_canonical: Optional[str], value: Optional[str]) -> Optional[str]:
    if not (l1 and l2_canonical and value):
        return None
    l2_label = l2_canonical.split('/', 1)[1]
    normalized = _norm(value)
    for candidate in LABEL_HIERARCHY.get(l1, {}).get(l2_label, []):
        if _norm(candidate) == normalized:
            return f"{l1}/{l2_label}/{candidate}"
    return None


def extract_label_targets(example: dict) -> Dict[str, Optional[int]]:
    hierarchy = example.get('hierarchy') or example.get('metadata', {}).get('hierarchy') or {}
    l1_candidate = example.get('l1') or hierarchy.get('l1')
    l2_candidate = example.get('l2') or hierarchy.get('l2')
    l3_candidate = example.get('l3') or hierarchy.get('l3')

    l1_canonical = canonicalize_l1(l1_candidate)
    l2_canonical = canonicalize_l2(l1_canonical, l2_candidate) if l1_canonical else None
    l3_canonical = canonicalize_l3(l1_canonical, l2_canonical, l3_candidate) if l2_canonical else None

    l1_idx = L1_TO_ID.get(l1_canonical) if l1_canonical else -100
    l2_idx = L2_TO_ID.get(l2_canonical) if l2_canonical else -100
    l3_idx = L3_TO_ID.get(l3_canonical) if l3_canonical else -100

    metadata = example.get('metadata', {})
    verification = metadata.get('verification', {})

    tb_type = example.get('tb_type') or verification.get('tb_type') or metadata.get('tb_type') or "template"
    tb_type_norm = _norm(tb_type) if tb_type else None
    tb_key = TB_TYPE_CANONICAL.get(tb_type_norm or "", "template")
    tb_idx = TB_TYPE_TO_ID[tb_key]

    formal_pass = bool(verification.get('formal_pass', metadata.get('formal_pass', False)))

    code_lower = (example.get('output') or "").lower()
    clock_present = 1 if re.search(r'\bclk(_i)?\b', code_lower) else 0
    reset_present = 1 if re.search(r'\brst_n\b|\brst\b|\breset(_n)?\b', code_lower) else 0
    output_present = 1 if re.search(r'\boutput\b', code_lower) else 0
    is_fsm = (l2_canonical == "Sequential/FSM")

    io_presence = [clock_present, reset_present, output_present]
    io_mask = 1 if is_fsm else 0

    return {
        "l1_idx": l1_idx,
        "l2_idx": l2_idx,
        "l3_idx": l3_idx,
        "tb_idx": tb_idx,
        "formal_pass": 1.0 if formal_pass else 0.0,
        "io_presence": io_presence,
        "io_mask": io_mask
    }


class RTLDatasetFormatter:
    """Format RTL dataset for instruction tuning"""

    def __init__(self, tokenizer, conditioning_metadata: bool = False):
        self.tokenizer = tokenizer
        self.conditioning_metadata = conditioning_metadata

    @staticmethod
    def _conditioning_tokens(example: dict) -> str:
        l1 = example.get('l1') or example.get('category')
        l2 = example.get('l2')
        l3 = example.get('l3')
        tokens = []
        if l1:
            tokens.append(f"DesignType:{l1}")
        if l2:
            tokens.append(f"Subtype:{l1}/{l2}")
        if l3:
            tokens.append(f"Variant:{l1}/{l2}/{l3}")
        if not tokens:
            return ""
        return "[" + ";".join(tokens) + "]\n"

    def format_example(self, example):
        """Format a single example for instruction tuning"""

        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']

        if self.conditioning_metadata:
            prefix = self._conditioning_tokens(example)
            if prefix:
                instruction = prefix + instruction

        # Qwen2.5-Coder Instruct format
        system_msg = "You are an expert RTL (Register Transfer Level) code generator. Generate clean, syntactically correct Verilog/SystemVerilog code."

        if input_text:
            user_msg = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_msg = instruction

        segments = [
            f"<|im_start|>system\n{system_msg}<|im_end|>\n",
            f"<|im_start|>user\n{user_msg}<|im_end|>\n",
            "<|im_start|>assistant\n",
            f"{output}<|im_end|>"
        ]

        segment_tokens: List[List[int]] = []
        for segment in segments:
            encoded = self.tokenizer(segment, add_special_tokens=False)
            segment_tokens.append(encoded['input_ids'])

        # Truncate if needed (trim response first)
        max_length = 2048
        total_tokens = sum(len(seg) for seg in segment_tokens)

        input_ids = list(sum(segment_tokens, []))
        attention_mask = [1] * len(input_ids)

        system_len = len(segment_tokens[0])
        user_len = len(segment_tokens[1])
        assistant_len = len(segment_tokens[2])
        prompt_len = system_len + user_len + assistant_len

        labels = [-100] * prompt_len + segment_tokens[3]
        labels = labels[:len(input_ids)]

        spec_mask = [0] * len(input_ids)
        # Mark user segment as spec tokens
        for idx in range(system_len, system_len + user_len):
            if idx < len(spec_mask):
                spec_mask[idx] = 1

        code_mask = [0] * len(input_ids)
        for idx in range(prompt_len, len(input_ids)):
            code_mask[idx] = 1

        if len(input_ids) > max_length:
            trim_start = len(input_ids) - max_length
            input_ids = input_ids[trim_start:]
            attention_mask = attention_mask[trim_start:]
            labels = labels[trim_start:]
            spec_mask = spec_mask[trim_start:]
            code_mask = code_mask[trim_start:]

        label_targets = extract_label_targets(example)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'spec_token_mask': spec_mask,
            'code_token_mask': code_mask,
            'l1_label': label_targets['l1_idx'],
            'l2_label': label_targets['l2_idx'],
            'l3_label': label_targets['l3_idx'],
            'tb_label': label_targets['tb_idx'],
            'formal_label': label_targets['formal_pass'],
            'formal_mask': 1,
            'io_presence': label_targets['io_presence'],
            'io_mask': label_targets['io_mask']
        }


class RTLDataCollator(DataCollatorForSeq2Seq):
    """Pad auxiliary masks alongside standard seq2seq features."""

    EXTRA_MASK_KEYS = ("spec_token_mask", "code_token_mask")

    def __call__(self, features, return_tensors=None):
        max_seq_len = max(len(f["input_ids"]) for f in features)
        for feat in features:
            current_len = len(feat["input_ids"])
            pad_len = max_seq_len - current_len
            if pad_len > 0:
                for key in self.EXTRA_MASK_KEYS:
                    if key in feat:
                        feat[key] = feat[key] + [0] * pad_len
        return super().__call__(features, return_tensors=return_tensors)


def attach_multitask_heads(model: nn.Module) -> None:
    """Register auxiliary classification heads used during fine-tuning."""
    hidden_size = model.config.hidden_size
    heads = nn.ModuleDict({
        'l1': nn.Linear(hidden_size, len(L1_LABELS)),
        'l2': nn.Linear(hidden_size, len(L2_LABELS)),
        'l3': nn.Linear(hidden_size, len(L3_LABELS)),
        'tb': nn.Linear(hidden_size, len(TB_TYPES)),
        'formal': nn.Linear(hidden_size, 1),
        'io': nn.Linear(hidden_size, 3)
    })
    heads.to(next(model.parameters()).device)
    model.add_module("multitask_heads", heads)


def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean hidden state for masked positions."""
    mask = mask.float().unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class MultiTaskTrainer(Trainer):
    """Custom trainer that adds auxiliary supervision losses."""

    def __init__(
        self,
        *args,
        lambda_label: float = 0.3,
        lambda_tb: float = 0.2,
        lambda_io: float = 0.2,
        lambda_formal: float = 0.1,
        lambda_infoce: float = 0.1,
        infoce_temperature: float = 0.07,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_label = lambda_label
        self.lambda_tb = lambda_tb
        self.lambda_io = lambda_io
        self.lambda_formal = lambda_formal
        self.lambda_infoce = lambda_infoce
        self.infoce_temperature = infoce_temperature
        self.extra_keys = {
            'l1_label',
            'l2_label',
            'l3_label',
            'tb_label',
            'formal_label',
            'formal_mask',
            'io_presence',
            'io_mask',
            'spec_token_mask',
            'code_token_mask'
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        extra = {k: inputs[k] for k in self.extra_keys if k in inputs}
        model_inputs = {k: v for k, v in inputs.items() if k not in self.extra_keys}

        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]
        attention_mask = model_inputs['attention_mask']
        pooled = _masked_mean(hidden_states, attention_mask)

        label_losses = []

        if 'l1_label' in extra:
            labels = extra['l1_label']
            valid = labels >= 0
            if valid.any():
                logits = model.multitask_heads['l1'](pooled[valid])
                label_losses.append(F.cross_entropy(logits, labels[valid]))

        if 'l2_label' in extra:
            labels = extra['l2_label']
            valid = labels >= 0
            if valid.any():
                logits = model.multitask_heads['l2'](pooled[valid])
                label_losses.append(F.cross_entropy(logits, labels[valid]))

        if 'l3_label' in extra:
            labels = extra['l3_label']
            valid = labels >= 0
            if valid.any():
                logits = model.multitask_heads['l3'](pooled[valid])
                label_losses.append(F.cross_entropy(logits, labels[valid]))

        if label_losses:
            loss = loss + self.lambda_label * torch.stack(label_losses).mean()

        if 'tb_label' in extra:
            labels = extra['tb_label']
            valid = labels >= 0
            if valid.any():
                logits = model.multitask_heads['tb'](pooled[valid])
                tb_loss = F.cross_entropy(logits, labels[valid])
                loss = loss + self.lambda_tb * tb_loss

        if 'formal_label' in extra:
            mask = extra.get('formal_mask')
            labels = extra['formal_label'].float()
            if mask is not None:
                valid = mask > 0
            else:
                valid = torch.ones_like(labels).bool()
            if valid.any():
                logits = model.multitask_heads['formal'](pooled[valid]).squeeze(-1)
                formal_loss = F.binary_cross_entropy_with_logits(logits, labels[valid])
                loss = loss + self.lambda_formal * formal_loss

        if 'io_presence' in extra:
            mask = extra.get('io_mask')
            labels = extra['io_presence'].float()
            if mask is not None:
                valid = mask > 0
            else:
                valid = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            if valid.any():
                logits = model.multitask_heads['io'](pooled[valid])
                io_loss = F.binary_cross_entropy_with_logits(logits, labels[valid])
                loss = loss + self.lambda_io * io_loss

        # Contrastive spec-code InfoNCE
        spec_mask = extra.get('spec_token_mask')
        code_mask = extra.get('code_token_mask')
        if spec_mask is not None and code_mask is not None:
            spec_lengths = spec_mask.sum(dim=1)
            code_lengths = code_mask.sum(dim=1)
            valid = (spec_lengths > 0) & (code_lengths > 0)
            if valid.sum() >= 2:
                spec_emb = _masked_mean(hidden_states[valid], spec_mask[valid])
                code_emb = _masked_mean(hidden_states[valid], code_mask[valid])
                spec_norm = F.normalize(spec_emb, dim=-1)
                code_norm = F.normalize(code_emb, dim=-1)
                logits = spec_norm @ code_norm.T / self.infoce_temperature
                targets = torch.arange(logits.size(0), device=logits.device)
                infoce_loss = 0.5 * (
                    F.cross_entropy(logits, targets) +
                    F.cross_entropy(logits.T, targets)
                )
                loss = loss + self.lambda_infoce * infoce_loss
                if model.training:
                    preds = logits.argmax(dim=-1)
                    false_neg_rate = (preds != targets).float().mean()
                    self.log({"train/false_neg_rate": false_neg_rate.item()})

        return (loss, outputs) if return_outputs else loss


def setup_model_and_tokenizer(model_name: str):
    """Load Qwen2.5-Coder model with 4-bit quantization"""

    print("\n🔧 Setting up Qwen2.5-Coder-7B model with QLoRA...")

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Nested quantization
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=64,  # LoRA rank
        lora_alpha=16,  # LoRA alpha
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Attach multi-task heads
    attach_multitask_heads(model)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Trainable Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer


def stage_filter(example: dict, stage: int) -> bool:
    if not stage or stage not in CATEGORY_STAGE_MAP:
        return True
    allowed = CATEGORY_STAGE_MAP[stage]
    if allowed is None:
        return True
    category = (example.get('category') or '').lower()
    l1 = (example.get('l1') or '').lower()
    if stage == 1:
        return category in allowed or l1 == 'combinational'
    if stage == 2:
        return category in allowed or l1 in {'sequential', 'combinational'}
    return True


def prepare_dataset(dataset_path: Path, tokenizer, curriculum_stage: int | None = None, conditioning_metadata: bool = False):
    """Load and prepare RTL dataset"""

    print(f"\n📂 Loading dataset: {dataset_path}")

    # Load JSONL dataset
    dataset = load_dataset('json', data_files=str(dataset_path), split='train')

    print(f"   Total examples: {len(dataset)}")

    if curriculum_stage:
        before = len(dataset)
        dataset = dataset.filter(lambda ex: stage_filter(ex, curriculum_stage))
        after = len(dataset)
        print(f"   Curriculum Stage {curriculum_stage}: {after}/{before} examples kept")
        if after == 0:
            raise ValueError(f"Curriculum stage {curriculum_stage} resulted in empty dataset.")

    # Split train/validation (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Format dataset
    formatter = RTLDatasetFormatter(tokenizer, conditioning_metadata=conditioning_metadata)

    train_dataset = dataset['train'].map(
        formatter.format_example,
        remove_columns=dataset['train'].column_names,
        desc="Formatting train set"
    )

    val_dataset = dataset['test'].map(
        formatter.format_example,
        remove_columns=dataset['test'].column_names,
        desc="Formatting validation set"
    )

    print(f"\n✅ Dataset prepared:")
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Validation: {len(val_dataset)} examples")

    return train_dataset, val_dataset


def train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: Path
):
    """Run QLoRA fine-tuning"""

    print("\n🚀 Starting QLoRA fine-tuning...")

    # Training arguments optimized for RTX 5090 (32GB VRAM) with Qwen2.5-Coder-7B
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increased for 8B model on 32GB
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",  # Memory-efficient optimizer
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=torch.cuda.is_available(),  # Only use bf16 when CUDA is available
        report_to="none",  # Disable tensorboard (not installed)
        logging_dir=str(output_dir / "logs"),
        push_to_hub=False,
        remove_unused_columns=False
    )

    # Data collator for padding
    data_collator = RTLDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # Trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        lambda_label=0.3,
        lambda_tb=0.2,
        lambda_io=0.2,
        lambda_formal=0.1,
        lambda_infoce=0.1
    )

    # Train
    print("\n⏱️  Training started...")
    start_time = datetime.now()

    trainer.train()

    duration = datetime.now() - start_time
    print(f"\n✅ Training complete! Duration: {duration}")

    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print(f"\n💾 Model saved to: {final_model_path}")

    # Save training stats
    stats = {
        'duration_seconds': duration.total_seconds(),
        'duration_human': str(duration),
        'train_examples': len(train_dataset),
        'val_examples': len(val_dataset),
        'final_train_loss': trainer.state.log_history[-1].get('loss', None),
        'final_eval_loss': trainer.state.log_history[-1].get('eval_loss', None)
    }

    stats_file = output_dir / "training_stats.json"
    with stats_file.open('w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n📊 Training statistics saved to: {stats_file}")


def main():
    """Main training pipeline"""

    print("="*80)
    print("  QWEN2.5-CODER-7B QLORA FINE-TUNING FOR RTL GENERATION")
    print("="*80)

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder on RTL data")
    parser.add_argument("--dataset", type=str, default="data/rtl_behavioral_v5_2.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--output-dir", type=str, help="Output directory for checkpoints")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--curriculum-stage", type=int, choices=[1, 2, 3], default=None, help="Curriculum stage filter (1=combinational, 2=sequential, 3=full)")
    parser.add_argument("--conditioning-metadata", action="store_true", help="Prepend hierarchy tokens to each instruction")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dataset_path = (project_root / args.dataset).resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output = project_root / 'models' / 'qwen_coder_rtl' / f"run_{dataset_path.stem}_{timestamp}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
    model_name = args.model

    print(f"\n📋 Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    if args.curriculum_stage:
        print(f"   Curriculum stage: {args.curriculum_stage}")
    print(f"   Conditioning metadata: {'ON' if args.conditioning_metadata else 'OFF'}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Check dataset exists
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("   Run prepare_rtl_dataset.py first!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(
        dataset_path,
        tokenizer,
        curriculum_stage=args.curriculum_stage,
        conditioning_metadata=args.conditioning_metadata
    )

    # Train
    train(model, tokenizer, train_dataset, val_dataset, output_dir)

    print("\n" + "="*80)
    print("  ✅ TRAINING COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
