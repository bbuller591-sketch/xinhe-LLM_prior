#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_qwen3_32b_measurements.py
Frozen local measurement runner for LOCAL_LLM_MEASUREMENT_BUNDLE_20260924.

Model   : /home/scc/pb24000232/model/Qwen3-32B  (BF16, non-thinking, greedy)
Method  : full-vocabulary logits at the first generated position. Semantic
          logsumexp over token ids whose tokenizer.decode([id]).strip() is
          A / B / U / T, then A-vs-B softmax normalization, binary entropy
          and certainty.
          Generated text is decoded greedily (argmax continuation, KV cache)
          purely for audit; measurements come from the first-step logits.

Usage:
    python run_qwen3_32b_measurements.py --mode preflight [--batch-size 4]
    python run_qwen3_32b_measurements.py --mode full [--batch-size 8] [--limit N]
    python run_qwen3_32b_measurements.py --mode status   # rebuild run_status.json offline

Resume safety: measurements.jsonl is appended one batch at a time with flush;
on startup already-completed portable_query_id rows are skipped.
"""
import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Frozen constants
# ----------------------------------------------------------------------------
WORKDIR = Path(__file__).resolve().parent
BUNDLE_DIR = WORKDIR / "LOCAL_LLM_MEASUREMENT_BUNDLE_20260924"
QUERIES_JSONL = BUNDLE_DIR / "queries" / "jsonl" / "all_queries.jsonl"
RETURN_OUTPUTS = WORKDIR / "return_outputs"
LOGS_DIR = RETURN_OUTPUTS / "logs"
MEASUREMENTS_JSONL = RETURN_OUTPUTS / "measurements.jsonl"
MODEL_RUNTIME_JSON = RETURN_OUTPUTS / "model_runtime.json"
RUN_STATUS_JSON = RETURN_OUTPUTS / "run_status.json"
SEM_GROUPS_JSON = RETURN_OUTPUTS / "tokenizer_semantic_groups.json"
AUDIT_DIR = WORKDIR / "audit"
PREFLIGHT_JSON = AUDIT_DIR / "QWEN3_32B_PREFLIGHT.json"

MODEL_PATH = "/home/scc/pb24000232/model/Qwen3-32B"
MODEL_ID = "Qwen3-32B"
DTYPE = torch.bfloat16
SEED = 20260924
DECODING_MODE = "greedy"          # argmax continuation; deterministic, no sampling
ENABLE_THINKING = False           # HARD CONSTRAINT: Qwen3 non-thinking mode
MAX_CONTEXT = 40960               # model max_position_embeddings; prompts max 3349
HOSPITAL_MAX_NEW = 220            # protocol original_max_tokens for Hospital
NORMAL_MAX_NEW = 4                # protocol original_max_tokens for A/B and A/B/U
EOS_TOKENS = {151645, 151643}     # <|im_end|>, <|endoftext|> per generation_config
THINK_MARKERS = ("<think", "think>", "/think", "<response", "response>")
EXPECTED_ROWS = 14542

# hashes of the frozen prompts are re-verified at startup against these fields
SEM_GROUP_LETTERS = ["A", "B", "U", "T"]


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def setup_logging(log_path):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("qwen3_runner")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


LOG = None  # set in main


class SystemicFailure(RuntimeError):
    """Stop-the-run error: model load, thinking contamination, hash mismatch,
    logits-extraction definition errors, truncation, or batch-1 OOM."""


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Semantic token groups (definition identical to bundle tools)
# ----------------------------------------------------------------------------
def build_semantic_groups(tokenizer):
    groups = {k: [] for k in SEM_GROUP_LETTERS}
    for i in range(len(tokenizer)):
        try:
            s = tokenizer.decode([i], skip_special_tokens=False)
        except Exception:
            continue
        z = s.strip()
        if z in groups:
            groups[z].append(i)
    LOG.info("semantic groups: %s",
             {k: len(v) for k, v in groups.items()})
    return groups


def load_semantic_groups(tokenizer):
    """Recompute groups (definition frozen in bundle tools) and cross-check
    against a cached copy if present."""
    groups = build_semantic_groups(tokenizer)
    if SEM_GROUPS_JSON.exists():
        cached = json.load(open(SEM_GROUPS_JSON, encoding="utf-8"))
        cached_ids = {k: [e["token_id"] for e in v] for k, v in
                      cached["semantic_token_groups"].items()}
        if cached_ids != groups:
            raise SystemicFailure(
                "semantic group mismatch vs cached tokenizer_semantic_groups.json")
    payload = {
        "model": MODEL_PATH,
        "vocab_size": len(tokenizer),
        "semantic_token_groups": {
            k: [{"token_id": i,
                 "decoded": tokenizer.decode([i], skip_special_tokens=False)}
                for i in v]
            for k, v in groups.items()
        },
    }
    RETURN_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(SEM_GROUPS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return groups


# ----------------------------------------------------------------------------
# Model / runtime
# ----------------------------------------------------------------------------
def gpu_info():
    info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": p.name,
            "memory_total_mib": p.total_memory // (1024 * 1024),
        })
    return info


def load_model_and_tokenizer():
    LOG.info("loading tokenizer %s", MODEL_PATH)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    LOG.info("loading model %s dtype=bfloat16", MODEL_PATH)
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        flash_available = True
    except ImportError:
        flash_available = False
    if flash_available:
        attn_impl = "flash_attention_2"
        LOG.info("flash-attn available -> using flash_attention_2")
    else:
        LOG.info("flash-attn NOT available -> using PyTorch SDPA (per protocol)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    devices = [str(d) for d in getattr(model, "hf_device_map", {}).values()] \
        or [str(model.device)]
    LOG.info("model loaded. devices=%s", devices)
    used = torch.cuda.max_memory_allocated(0) // (1024 ** 2)
    LOG.info("CUDA memory allocated after load: %d MiB", used)

    groups = load_semantic_groups(tok)
    return model, tok, groups, attn_impl, flash_available


def model_file_inventory():
    """File list + sizes; embed SHA256s if the background hash job finished."""
    files = []
    for p in sorted(Path(MODEL_PATH).iterdir()):
        if p.is_file():
            files.append({"name": p.name, "size_bytes": p.stat().st_size})
    sha_file = WORKDIR / "model_safetensors_sha256.txt"
    if sha_file.exists():
        shas = {}
        for line in sha_file.read_text(encoding="utf-8").splitlines():
            h, name = line.split(None, 1)
            shas[name.strip()] = h
        for f in files:
            if f["name"] in shas:
                f["sha256"] = shas[f["name"]]
    return files


def write_model_runtime(model, tok, groups, attn_impl, flash_available):
    cfg = model.config.to_dict()
    keep_cfg = {k: cfg[k] for k in
                ("architectures", "hidden_size", "num_hidden_layers",
                 "num_attention_heads", "num_key_value_heads",
                 "intermediate_size", "vocab_size", "max_position_embeddings",
                 "rope_theta", "sliding_window", "use_sliding_window",
                 "torch_dtype", "model_type") if k in cfg}
    runtime = OrderedDict([
        ("model_id", MODEL_ID),
        ("model_path", MODEL_PATH),
        ("architecture", cfg.get("architectures", ["Qwen3ForCausalLM"])[0]),
        ("dtype", "torch.bfloat16"),
        ("quantization", "none"),
        ("attention_implementation", attn_impl),
        ("flash_attn_available", flash_available),
        ("decoding_mode", DECODING_MODE),
        ("sampling", "disabled (greedy argmax continuation)"),
        ("seed", SEED),
        ("enable_thinking", ENABLE_THINKING),
        ("chat_template_application",
         "tokenizer.apply_chat_template(messages, tokenize=False, "
         "add_generation_prompt=True, enable_thinking=False); local "
         "template implements enable_thinking=False as a closed empty "
         "<think></think> block appended to the assistant prefix, so "
         "generation starts directly in non-thinking response mode"),
        ("gpu", gpu_info()),
        ("cuda_runtime_version", torch.version.cuda),
        ("torch_version", torch.__version__),
        ("transformers_version", __import__("transformers").__version__),
        ("python_version", sys.version.split()[0]),
        ("model_config", keep_cfg),
        ("tokenizer_config", {
            "vocab_size": len(tok),
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "padding_side": tok.padding_side,
            "chat_template_present": bool(tok.chat_template),
            "chat_template_sha256": sha_text(tok.chat_template or ""),
        }),
        ("model_files", model_file_inventory()),
        ("semantic_token_groups", {k: v for k, v in groups.items()}),
        ("context_truncation_allowed", False),
        ("max_context", MAX_CONTEXT),
        ("runner_sha256", sha_text(Path(__file__).read_text(encoding="utf-8"))),
    ])
    RETURN_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(MODEL_RUNTIME_JSON, "w", encoding="utf-8") as f:
        json.dump(runtime, f, ensure_ascii=False, indent=2)
    LOG.info("model_runtime.json written")
    return runtime


# ----------------------------------------------------------------------------
# Query loading + hash verification
# ----------------------------------------------------------------------------
def load_queries():
    rows = []
    for line in open(QUERIES_JSONL, encoding="utf-8"):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if len(rows) != EXPECTED_ROWS:
        raise SystemicFailure(f"query row count {len(rows)} != {EXPECTED_ROWS}")
    return rows


def verify_prompt_hashes(rows):
    bad = []
    for r in rows:
        if sha_text(r["system_prompt"]) != r["system_prompt_sha256"]:
            bad.append(("system", r["portable_query_id"]))
        if sha_text(r["user_prompt"]) != r["user_prompt_sha256"]:
            bad.append(("user", r["portable_query_id"]))
    if bad:
        raise SystemicFailure(
            f"prompt hash mismatch on {len(bad)} rows (first: {bad[:5]})")
    LOG.info("prompt SHA256 verified for all %d rows", len(rows))


def build_template_text(tok, row):
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_prompt"]},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING)
    return text


# ----------------------------------------------------------------------------
# Hospital structured-output parser (frozen protocol SYSTEM_HOSPITAL.txt)
# ----------------------------------------------------------------------------
def parse_hospital(text, user_prompt):
    allowed_ids = set()
    for m in re.finditer(r"\b([AB])\.(E\d+)\b", user_prompt):
        allowed_ids.add(f"{m.group(1)}.{m.group(2)}")
    issues = []
    first_tok = None
    m_first = re.match(r"^\s*([AB])(?=[\s\.,;:!?\n]|$)", text)
    if not m_first:
        issues.append("first_token_not_AB")
    else:
        first_tok = m_first.group(1)
    lines = {}
    for key in ("CITES_A", "CITES_B", "RATIONALE"):
        m = re.search(rf"(?m)^{key}\s*=\s*(.*)$", text)
        lines[key] = m.group(1).strip() if m else None
    cited = {}
    for key in ("CITES_A", "CITES_B"):
        if lines[key] is None:
            issues.append(f"missing_{key}")
            cited[key] = None
            continue
        val = lines[key]
        if val.upper() == "NONE":
            cited[key] = []
            continue
        ids = [x.strip() for x in val.split(",") if x.strip()]
        cited[key] = ids
        if not ids:
            issues.append(f"empty_{key}")
        for cid in ids:
            if not re.match(r"^[AB]\.E\d+$", cid):
                issues.append(f"bad_cite_format_{key}:{cid}")
            elif cid not in allowed_ids:
                issues.append(f"cite_not_in_prompt_{key}:{cid}")
    if lines["RATIONALE"] is None:
        issues.append("missing_RATIONALE")
    status = "PASS" if not issues else "FAIL_HOSPITAL_" + "|".join(issues[:4])
    return {
        "first_token": first_tok,
        "cites_a": cited["CITES_A"],
        "cites_b": cited["CITES_B"],
        "rationale": lines["RATIONALE"],
        "allowed_evidence_ids_in_prompt": sorted(allowed_ids),
        "hospital_parser_status": status,
    }


# ----------------------------------------------------------------------------
# Inference engine
# ----------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, model, tok, groups, batch_size):
        self.model = model
        self.tok = tok
        self.groups = groups
        self.device = next(model.parameters()).device
        self.batch_size = batch_size
        self.oom_events = 0
        self.batch_size_history = [batch_size]
        # group index tensors on device
        self.group_idx = {k: torch.tensor(v, dtype=torch.long, device=self.device)
                          for k, v in groups.items()}
        self.stop_ids = sorted(EOS_TOKENS)

    def run_batch(self, rows, max_new):
        """Run one batch. Returns list of result dicts (unpersisted)."""
        texts = [r["_template_text"] for r in rows]
        B = len(rows)
        t0 = time.time()
        enc = self.tok(texts, padding=True, truncation=False,
                       return_tensors="pt")
        inp = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device)
        n_tok = int(am.sum().item())
        if inp.shape[1] > MAX_CONTEXT:
            raise SystemicFailure(
                f"batch input length {inp.shape[1]} exceeds context {MAX_CONTEXT}")

        with torch.inference_mode():
            out = self.model(input_ids=inp, attention_mask=am, use_cache=True)
            first_logits = out.logits[:, -1, :]          # [B, V] first generated position
            if not torch.isfinite(first_logits).all():
                raise SystemicFailure("non-finite logits in first step (systemic)")
            cur = first_logits.argmax(dim=-1)
            past = out.past_key_values
            finished = torch.zeros(B, dtype=torch.bool, device=self.device)
            gen_ids = [[] for _ in range(B)]
            for _step in range(max_new):
                for i in range(B):
                    if not finished[i].item():
                        t = cur[i].item()
                        gen_ids[i].append(t)
                        if t in self.stop_ids:
                            finished[i] = True
                if bool(finished.all().item()):
                    break
                nxt = torch.where(finished, torch.full_like(
                    cur, self.tok.pad_token_id), cur)
                out = self.model(input_ids=nxt.unsqueeze(1),
                                 past_key_values=past, use_cache=True)
                past = out.past_key_values
                cur = out.logits[:, -1, :].argmax(dim=-1)

        results = []
        elapsed = time.time() - t0
        for i, row in enumerate(rows):
            results.append(self._build_result(
                row, first_logits[i], gen_ids[i], elapsed / B))
        return results, n_tok, elapsed

    def _build_result(self, row, logits_v, gen_ids, runtime_sec):
        tok = self.tok
        V = logits_v.shape[0]
        lA = float(torch.logsumexp(logits_v[self.group_idx["A"]], dim=0).item())
        lB = float(torch.logsumexp(logits_v[self.group_idx["B"]], dim=0).item())
        lU = float(torch.logsumexp(logits_v[self.group_idx["U"]], dim=0).item())
        lT = float(torch.logsumexp(logits_v[self.group_idx["T"]], dim=0).item())
        # A-vs-B normalization in float64 for numerical safety
        ab = torch.tensor([lA, lB], dtype=torch.float64)
        pA, pB = [float(x) for x in torch.softmax(ab, dim=0).tolist()]
        if pA in (0.0, 1.0):
            H = 0.0
        else:
            H = float(-pA * np.log2(pA) - pB * np.log2(pB))
        certainty = 1.0 - H

        topk_vals, topk_ids = torch.topk(logits_v, k=5)
        top5 = [{"token_id": int(tid), "logit": float(lv),
                 "decoded": tok.decode([int(tid)], skip_special_tokens=False)}
                for lv, tid in zip(topk_vals.tolist(), topk_ids.tolist())]
        argmax_id = int(topk_ids[0].item())

        # decode generated ids, cut at first stop token
        cut = []
        for t in gen_ids:
            cut.append(t)
            if t in EOS_TOKENS:
                break
        text_raw = tok.decode(cut, skip_special_tokens=False)
        text_clean = tok.decode(cut, skip_special_tokens=True).strip()
        first_gen_id = cut[0] if cut else None
        first_gen_tok = (tok.decode([first_gen_id], skip_special_tokens=False).strip()
                         if first_gen_id is not None else "")

        think_contamination = any(mk in text_raw for mk in THINK_MARKERS)
        if think_contamination:
            raise SystemicFailure(
                f"thinking contamination detected for {row['portable_query_id']}: "
                f"{text_raw[:200]!r}")

        allowed = set(row["allowed_tokens"])
        argmax_dec = tok.decode([argmax_id], skip_special_tokens=False).strip()
        in_contract = argmax_dec in allowed

        result = OrderedDict([
            ("portable_query_id", row["portable_query_id"]),
            ("original_query_id", row["original_query_id"]),
            ("bundle_id", row["bundle_id"]),
            ("dataset", row["dataset"]),
            ("phase", row["phase"]),
            ("model_id", MODEL_ID),
            ("model_path", MODEL_PATH),
            ("generated_text", text_clean),
            ("generated_text_raw_with_specials", text_raw),
            ("first_generated_token", first_gen_tok),
            ("semantic_logit_A", lA),
            ("semantic_logit_B", lB),
            ("semantic_logit_U", lU),
            ("semantic_logit_T", lT),
            ("pA_vs_B", pA),
            ("pB_vs_A", pB),
            ("entropy_AB_bits", H),
            ("certainty", certainty),
            ("input_token_count", 0),   # filled by caller
            ("generation_token_count", len(cut)),
            ("runtime_seconds", runtime_sec),
            ("status", "PASS"),
        ])
        extra = OrderedDict([
            ("argmax_token_id", argmax_id),
            ("argmax_decoded_token", argmax_dec),
            ("semantic_group_ids",
             {k: [int(x) for x in self.group_idx[k].tolist()] for k in SEM_GROUP_LETTERS}),
            ("raw_first_step_top5", top5),
            ("system_prompt_sha256", row["system_prompt_sha256"]),
            ("user_prompt_sha256", row["user_prompt_sha256"]),
            ("hash_match", True),
            ("thinking_contamination", False),
            ("truncated", False),
            ("allowed_tokens", sorted(allowed)),
            ("decoding_mode", DECODING_MODE),
            ("vocab_dim", V),
        ])
        if row["dataset"] == "Hospital Osteoporosis":
            hp = parse_hospital(text_clean, row["user_prompt"])
            extra["hospital_parser"] = hp
            if hp["hospital_parser_status"] != "PASS":
                result["status"] = hp["hospital_parser_status"]
        result.update(extra)
        return result

    def run_queries(self, rows, mode, log, limit=None, progress_every=25):
        """Process rows (resume-safe handled by caller via completed set)."""
        # separate hospital (batch=1) from normal queries
        hospital = [r for r in rows if r["dataset"] == "Hospital Osteoporosis"]
        normal = [r for r in rows if r["dataset"] != "Hospital Osteoporosis"]
        order = [(hospital, HOSPITAL_MAX_NEW, 1)] + \
                [(normal, NORMAL_MAX_NEW, self.batch_size)]
        done = 0
        total = len(rows)
        start = time.time()
        for group_rows, max_new, bs in order:
            if not group_rows:
                continue
            # sort by prompt token length for tighter padding within batches
            group_rows = sorted(group_rows, key=lambda r: r["_tok_len"])
            batches = [group_rows[i:i + bs] for i in range(0, len(group_rows), bs)]
            idx = 0
            while idx < len(batches):
                batch = batches[idx]
                t0 = time.time()
                try:
                    results, n_tok, elapsed = self.run_batch(batch, max_new)
                except torch.cuda.OutOfMemoryError:
                    self.oom_events += 1
                    torch.cuda.empty_cache()
                    new_bs = max(1, len(batch) // 2)
                    if new_bs == len(batch):
                        raise SystemicFailure("OOM at batch size 1")
                    log.warning("OOM on %d-query batch -> re-splitting to %d (event %d)",
                                len(batch), new_bs, self.oom_events)
                    self.batch_size = new_bs
                    self.batch_size_history.append(new_bs)
                    # re-split current batch + remaining batches at new size
                    rest = [r for b in batches[idx:] for r in b]
                    batches = [rest[i:i + new_bs]
                               for i in range(0, len(rest), new_bs)]
                    idx = 0
                    continue
                for r, res in zip(batch, results):
                    r["_result"] = res
                    res["input_token_count"] = r["_tok_len"]
                    res["run_phase"] = mode
                yield batch, results
                done += len(batch)
                if done % progress_every == 0 or done == total:
                    el = time.time() - start
                    tps = n_tok / max(elapsed, 1e-6)
                    log.info(
                        "[%s] %d/%d queries done | batch %d | %d tok in %.1fs "
                        "(%.0f tok/s) | elapsed %.1f min | ETA %.1f min | OOM %d",
                        mode, done, total, len(batch), n_tok, elapsed, tps,
                        el / 60, (el / max(done, 1)) * (total - done) / 60,
                        self.oom_events)
                idx += 1


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def load_completed_ids():
    completed = set()
    if MEASUREMENTS_JSONL.exists():
        with open(MEASUREMENTS_JSONL, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                completed.add(r["portable_query_id"])
    return completed


def append_results(results):
    RETURN_OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(MEASUREMENTS_JSONL, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
def run_preflight(args):
    LOG.info("===== PREFLIGHT =====")
    rows = load_queries()
    verify_prompt_hashes(rows)
    model, tok, groups, attn_impl, flash_available = load_model_and_tokenizer()
    runtime = write_model_runtime(model, tok, groups, attn_impl, flash_available)

    # ---- template-level thinking check ----
    # NOTE: this local tokenizer carries a customized chat template in which
    #   enable_thinking=False  -> appends a CLOSED EMPTY think block
    #       '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    #   so generation starts AFTER the think phase is closed (non-thinking).
    #   enable_thinking=True/undefined -> no markers; the model may freely
    #   enter its default (thinking) mode.
    # The hard constraint is that GENERATED text has no thinking prefix; this
    # is enforced per-query in InferenceEngine._build_result (systemic abort).
    sample_row = rows[0]
    text_off = build_template_text(tok, sample_row)
    text_on = tok.apply_chat_template(
        [{"role": "system", "content": sample_row["system_prompt"]},
         {"role": "user", "content": sample_row["user_prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True)
    LOG.info("template OFF suffix: %r", text_off[-60:])
    LOG.info("template ON  suffix: %r", text_on[-60:])

    # ---- stratified sample: 4 rows per required dataset ----
    datasets_needed = ["Renal TCMR", "GSE272769", "Breast GSE25055->GSE25065",
                       "CREDIT-G", "Hospital Osteoporosis", "Darmanis GBM"]
    sample = []
    seen = Counter()
    for r in rows:
        ds = r["dataset"]
        if ds in datasets_needed and seen[ds] < 4:
            sample.append(r)
            seen[ds] += 1
    LOG.info("preflight sample: %d queries across %s",
             len(sample), {k: v for k, v in seen.items()})

    for r in rows:
        r["_template_text"] = build_template_text(tok, r)
        r["_tok_len"] = len(tok.encode(r["_template_text"]))
    for r in sample:
        if r["_tok_len"] > MAX_CONTEXT:
            raise SystemicFailure(f"preflight query exceeds context: {r['portable_query_id']}")

    engine = InferenceEngine(model, tok, groups, args.batch_size)
    # resume-safe: rows already measured (from an earlier preflight attempt)
    # are loaded back instead of re-inferred
    completed_ids = load_completed_ids()
    existing = {}
    if MEASUREMENTS_JSONL.exists():
        for line in open(MEASUREMENTS_JSONL, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                if x["portable_query_id"] in {r["portable_query_id"] for r in sample}:
                    existing[x["portable_query_id"]] = x
    sample_todo = [r for r in sample if r["portable_query_id"] not in completed_ids]
    LOG.info("preflight: %d already measured, %d to run",
             len(sample) - len(sample_todo), len(sample_todo))
    collected = []
    with torch.inference_mode():
        for batch, results in engine.run_queries(sample_todo, "preflight", LOG):
            append_results(results)
            collected.extend(results)
    for r in sample:
        if r["portable_query_id"] not in {x["portable_query_id"] for x in collected}:
            collected.append(existing[r["portable_query_id"]])
    collected.sort(key=lambda x: x["portable_query_id"])

    # ---- greedy manual-loop vs model.generate cross-check (2 queries) ----
    gen_crosscheck = []
    for r in sample[:2]:
        with torch.inference_mode():
            enc = tok([r["_template_text"]], padding=False, truncation=False,
                      return_tensors="pt").to(engine.device)
            out = model.generate(enc["input_ids"], max_new_tokens=NORMAL_MAX_NEW,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
        mine = next(x["generated_text"] for x in collected
                    if x["portable_query_id"] == r["portable_query_id"])
        gen_crosscheck.append({"portable_query_id": r["portable_query_id"],
                               "match": txt == mine,
                               "generate": txt, "manual_loop": mine})
        LOG.info("generate() crosscheck %s match=%s", r["portable_query_id"], txt == mine)
    if not all(g["match"] for g in gen_crosscheck):
        raise SystemicFailure(f"manual greedy loop disagrees with model.generate: {gen_crosscheck}")

    # ---- sanity checks ----
    hospital_rows = [x for x in collected if x["dataset"] == "Hospital Osteoporosis"]
    normal_rows = [x for x in collected if x["dataset"] != "Hospital Osteoporosis"]
    checks = OrderedDict([
        ("sample_n", len(collected)),
        ("enable_thinking_false_mechanism",
         text_off.endswith("<think>\n\n</think>\n\n")),
        ("A_group_nonempty", len(groups["A"]) > 0),
        ("B_group_nonempty", len(groups["B"]) > 0),
        ("U_group_nonempty", len(groups["U"]) > 0),
        ("semantic_logits_all_finite",
         bool(np.all(np.isfinite(
             [[x["semantic_logit_A"], x["semantic_logit_B"],
               x["semantic_logit_U"], x["semantic_logit_T"]]
              for x in collected])))),
        ("pA_plus_pB_close_to_1",
         all(abs(x["pA_vs_B"] + x["pB_vs_A"] - 1.0) < 1e-9 for x in collected)),
        ("entropy_in_01",
         all(0.0 - 1e-9 <= x["entropy_AB_bits"] <= 1.0 + 1e-9 for x in collected)),
        ("certainty_in_01",
         all(0.0 - 1e-9 <= x["certainty"] <= 1.0 + 1e-9 for x in collected)),
        ("prompt_hashes_all_match", all(x["hash_match"] for x in collected)),
        ("no_truncation", all(not x["truncated"] for x in collected)),
        ("generated_texts_have_no_think_markers",
         all(not x["thinking_contamination"] for x in collected)),
        ("generated_first_tokens_clean",
         all(x["first_generated_token"] in x["allowed_tokens"]
             for x in normal_rows)),
        ("hospital_parser_executed",
         all("hospital_parser" in x for x in hospital_rows)),
        ("normal_argmax_in_contract_ratio",
         sum(1 for x in normal_rows
             if x["argmax_decoded_token"] in x["allowed_tokens"]) / max(1, len(normal_rows))),
        ("greedy_crosscheck_match", all(g["match"] for g in gen_crosscheck)),
    ])
    passed = all(bool(v) for v in checks.values())
    LOG.info("preflight checks: %s", json.dumps(dict(checks), indent=2))
    LOG.info("preflight overall: %s", "PASS" if passed else "FAIL")

    preflight = {
        "run": "QWEN3_32B_PREFLIGHT",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_id": MODEL_ID,
        "model_path": MODEL_PATH,
        "sample_portable_query_ids": [x["portable_query_id"] for x in collected],
        "datasets_covered": sorted({x["dataset"] for x in collected}),
        "checks": checks,
        "overall": "PASS" if passed else "FAIL",
        "greedy_crosscheck": gen_crosscheck,
        "template_off_suffix": text_off[-80:],
        "template_on_suffix": text_on[-80:],
        "template_note": ("Local tokenizer carries a customized chat template: "
                          "enable_thinking=False appends a closed empty "
                          "<think></think> block so generation starts in "
                          "non-thinking response mode; enable_thinking=True "
                          "adds no markers (model may think by default)."),
        "runtime_summary": runtime,
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFLIGHT_JSON, "w", encoding="utf-8") as f:
        json.dump(preflight, f, ensure_ascii=False, indent=2)
    LOG.info("preflight written to %s", PREFLIGHT_JSON)
    if not passed:
        raise SystemicFailure("preflight FAILED - stopping before full run")
    LOG.info("===== PREFLIGHT PASSED =====")
    return 0


# ----------------------------------------------------------------------------
# Full run
# ----------------------------------------------------------------------------
def run_full(args):
    LOG.info("===== FULL RUN =====")
    rows = load_queries()
    verify_prompt_hashes(rows)
    model, tok, groups, attn_impl, flash_available = load_model_and_tokenizer()
    write_model_runtime(model, tok, groups, attn_impl, flash_available)

    completed = load_completed_ids()
    todo = [r for r in rows if r["portable_query_id"] not in completed]
    LOG.info("resume: %d already completed, %d remaining", len(completed), len(todo))
    if args.limit:
        todo = todo[:args.limit]
        LOG.info("limit set: processing first %d rows", len(todo))

    for r in todo:
        r["_template_text"] = build_template_text(tok, r)
        r["_tok_len"] = len(tok.encode(r["_template_text"]))
        if r["_tok_len"] > MAX_CONTEXT:
            raise SystemicFailure(
                f"query exceeds context {MAX_CONTEXT}: {r['portable_query_id']} "
                f"({r['_tok_len']} tokens)")

    engine = InferenceEngine(model, tok, groups, args.batch_size)
    t_start = time.time()
    n_done = len(completed)
    total = len(rows)
    for batch, results in engine.run_queries(todo, "full", LOG):
        append_results(results)
        n_done += len(batch)
    LOG.info("full run finished: %d/%d total rows measured in %.1f min",
             n_done, total, (time.time() - t_start) / 60)
    build_run_status(engine)
    return 0


# ----------------------------------------------------------------------------
# run_status.json
# ----------------------------------------------------------------------------
def build_run_status(engine=None):
    rows = []
    if MEASUREMENTS_JSONL.exists():
        for line in open(MEASUREMENTS_JSONL, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    ids = [r["portable_query_id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    all_rows = load_queries()
    all_ids = {r["portable_query_id"] for r in all_rows}
    missing = sorted(all_ids - set(ids))
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] != "PASS")
    hospital = [r for r in rows if r["dataset"] == "Hospital Osteoporosis"]
    hospital_fail = [r for r in hospital if r["status"] != "PASS"]
    normal = [r for r in rows if r["dataset"] != "Hospital Osteoporosis"]
    argmax_ok = sum(1 for r in normal
                    if r["argmax_decoded_token"] in r["allowed_tokens"])
    pA = [r["pA_vs_B"] for r in rows]
    cert = [r["certainty"] for r in rows]
    itc = [r["input_token_count"] for r in rows]
    hist = lambda xs, n=10: (
        {"min": float(min(xs)), "max": float(max(xs)),
         "mean": float(np.mean(xs)), "median": float(np.median(xs)),
         "p25": float(np.percentile(xs, 25)), "p75": float(np.percentile(xs, 75))}
        if xs else None)

    def first_tok_dist(subset):
        c = Counter(r["first_generated_token"] for r in subset)
        return dict(c.most_common(20))

    status = OrderedDict([
        ("run", "QWEN3_32B_LOCAL_MEASUREMENT_20260924"),
        ("generated_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
        ("expected", EXPECTED_ROWS),
        ("completed", len(rows)),
        ("unique_portable_query_ids", len(set(ids))),
        ("PASS", n_pass),
        ("FAIL", n_fail),
        ("missing", len(missing)),
        ("missing_ids", missing[:20]),
        ("duplicated_ids", dup),
        ("by_dataset", dict(Counter(r["dataset"] for r in rows))),
        ("by_phase", dict(Counter(r["phase"] for r in rows))),
        ("by_bundle", dict(Counter(r["bundle_id"] for r in rows))),
        ("first_token_distribution_normal_queries", first_tok_dist(normal)),
        ("first_token_distribution_hospital", first_tok_dist(hospital)),
        ("normal_argmax_in_allowed_tokens_ratio",
         argmax_ok / max(1, len(normal))),
        ("pA_vs_B_stats", hist(pA)),
        ("certainty_stats", hist(cert)),
        ("input_token_count_stats", hist(itc)),
        ("generation_token_count_stats", hist([r["generation_token_count"] for r in rows])),
        ("runtime_total_seconds", float(sum(r["runtime_seconds"] for r in rows))),
        ("runtime_avg_per_query_seconds",
         float(np.mean([r["runtime_seconds"] for r in rows])) if rows else None),
        ("oom_events", engine.oom_events if engine else None),
        ("batch_size_history", engine.batch_size_history if engine else None),
        ("truncated_any", any(r["truncated"] for r in rows)),
        ("thinking_contamination_any", any(r["thinking_contamination"] for r in rows)),
        ("hash_mismatch_any", any(not r["hash_match"] for r in rows)),
        ("hospital_total", len(hospital)),
        ("hospital_format_failures", len(hospital_fail)),
        ("hospital_failure_details",
         [{"portable_query_id": r["portable_query_id"],
           "status": r["status"]} for r in hospital_fail]),
        ("measurements_sha256", sha_text(Path(MEASUREMENTS_JSONL).read_text(encoding="utf-8"))
         if MEASUREMENTS_JSONL.exists() else None),
    ])
    with open(RUN_STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    LOG.info("run_status.json written: expected=%d completed=%d PASS=%d FAIL=%d "
             "missing=%d dup=%d", status["expected"], status["completed"],
             status["PASS"], status["FAIL"], status["missing"],
             len(status["duplicated_ids"]))
    return status


# ----------------------------------------------------------------------------
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["preflight", "full", "status"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit remaining queries in full mode (debug only)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    log_path = LOGS_DIR / ({"preflight": "preflight.log",
                            "full": "full_run.log",
                            "status": "status.log"}[args.mode])
    LOG = setup_logging(log_path)
    LOG.info("runner started mode=%s batch_size=%d seed=%d", args.mode,
             args.batch_size, SEED)
    try:
        if args.mode == "preflight":
            return run_preflight(args)
        if args.mode == "full":
            return run_full(args)
        if args.mode == "status":
            build_run_status(None)
            return 0
    except SystemicFailure as e:
        LOG.error("SYSTEMIC FAILURE: %s", e)
        LOG.error(traceback.format_exc())
        return 2
    except Exception as e:
        LOG.error("unexpected error: %s", e)
        LOG.error(traceback.format_exc())
        return 3


if __name__ == "__main__":
    sys.exit(main())
