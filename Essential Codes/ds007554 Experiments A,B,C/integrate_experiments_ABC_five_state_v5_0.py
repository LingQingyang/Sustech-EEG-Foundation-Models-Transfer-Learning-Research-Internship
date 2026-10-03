#!/usr/bin/env python3
"""Audit and aggregate the integrated five-state A/B/C v5.0 pipeline.

C's primary slow/fast/PCA/ambient comparisons are constructionally aligned inside
B's fold-local ledger. Experiment A remains an external endpoint audit. Therefore
B<->C ledger mismatches are fatal, while A skips or A<->B ledger differences are
reported explicitly without invalidating C's internal scientific comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

SCRIPT_VERSION = "2026-07-30-integrate-ABC-five-state-v5.0-fixed-k-audit"
ARMS = ("F0_LD", "F1_QD", "F2_LD_plus_QResidual", "F3_Tensor")
TOLERATED_STATUSES = {"pass", "C_skipped", "A_skipped", "A_missing", "A_ledger_warning"}


def now() -> str: return time.strftime("%Y-%m-%d %H:%M:%S")
def ensure_dir(path: Path | str) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def decode(v: Any) -> Any:
    if isinstance(v, np.generic): return v.item()
    if isinstance(v, np.ndarray): return [decode(x) for x in v.tolist()]
    if isinstance(v, Path): return str(v)
    if isinstance(v, Mapping): return {str(k): decode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [decode(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def json_dump(path: Path | str, payload: Any) -> None:
    path = Path(path); ensure_dir(path.parent); tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f: json.dump(decode(payload), f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def csv_read(path: Path | str) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0: return []
    with path.open("r", encoding="utf-8", newline="") as f: return list(csv.DictReader(f))

def csv_write(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path); ensure_dir(path.parent); rows = list(rows)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                v = decode(row.get(key, ""))
                if isinstance(v, (list, dict)): v = json.dumps(v, ensure_ascii=False)
                out[key] = v
            w.writerow(out)
    tmp.replace(path)

def as_float(v: Any) -> float:
    try:
        if v is None or str(v).strip() == "": return float("nan")
        return float(v)
    except (TypeError, ValueError): return float("nan")

def finite_mean(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float); a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")

def finite_sem(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=float); a = a[np.isfinite(a)]
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def require_prefix(value: str, prefix: str) -> str:
    """Return *value* without *prefix*, with Python 3.8 compatibility."""
    if not value.startswith(prefix):
        raise ValueError(f"Expected path component starting with {prefix!r}, got {value!r}")
    return value[len(prefix):]


def fold_key_from_c_dir(c_dir: Path) -> Tuple[str, int, str, int, int]:
    # .../folds/MODEL/sub-XXX/grid-G/heldout-session-Y/svd-rank-Z
    return (
        c_dir.parents[3].name,
        int(require_prefix(c_dir.parents[2].name, "sub-")),
        require_prefix(c_dir.parents[1].name, "grid-"),
        int(require_prefix(c_dir.parent.name, "heldout-session-")),
        int(require_prefix(c_dir.name, "svd-rank-")),
    )

def a_dir(a_root: Path, key: Tuple[str, int, str, int, int]) -> Path:
    m, s, _g, h, r = key
    return a_root / "folds" / m / f"sub-{s:03d}" / f"heldout-session-{h}" / f"svd-rank-{r}"

def b_dir(b_root: Path, key: Tuple[str, int, str, int, int]) -> Path:
    m, s, g, h, r = key
    return b_root / m / f"sub-{s:03d}" / f"grid-{g}" / f"heldout-session-{h}" / f"svd-rank-{r}"


def canonical_csv_ledger(path: Path) -> Dict[str, np.ndarray]:
    rows = sorted(csv_read(path), key=lambda x: int(x["global_index"]))
    fields = ["global_index", "session_id", "run_id", "sample_start", "sample_end", "phase_id", "task_id", "state_label"]
    return {f: np.asarray([int(r[f]) for r in rows], dtype=np.int64) for f in fields}

def canonical_b_ledger(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        held = np.asarray(z["is_heldout_session"], dtype=bool)
        order = np.argsort(np.asarray(z["global_index"], dtype=np.int64)[held])
        mapping = {
            "global_index":"global_index", "session_id":"session_id", "run_id":"run_id",
            "sample_start":"sample_start", "sample_end":"sample_end", "phase_id":"phase_id",
            "task_id":"task_id", "state_label":"state_label",
        }
        return {k: np.asarray(z[v], dtype=np.int64)[held][order] for k, v in mapping.items()}

def compare_ledgers(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    nl = len(next(iter(left.values()))) if left else 0; nr = len(next(iter(right.values()))) if right else 0
    if nl != nr: return {"status":"row_count_mismatch", "lengths":{"left":nl,"right":nr}, "mismatch_counts":{}}
    fields = sorted(set(left) & set(right)); mismatch = {f:int(np.sum(left[f] != right[f])) for f in fields}
    return {"status":"pass" if max(mismatch.values(), default=0)==0 else "value_mismatch", "lengths":{"left":nl,"right":nr}, "mismatch_counts":mismatch}

def run_level_index_diff(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
    def bucket(x: Mapping[str, np.ndarray]) -> Dict[Tuple[int,int], set[int]]:
        out: Dict[Tuple[int,int], set[int]] = {}
        for run, task, idx in zip(x["run_id"], x["task_id"], x["global_index"]):
            out.setdefault((int(run), int(task)), set()).add(int(idx))
        return out
    a, b = bucket(left), bucket(right); rows=[]
    for key in sorted(set(a)|set(b)):
        only_a=sorted(a.get(key,set())-b.get(key,set())); only_b=sorted(b.get(key,set())-a.get(key,set()))
        if only_a or only_b:
            rows.append({"run_id":key[0],"task_id":key[1],"n_A_only":len(only_a),"n_B_only":len(only_b),"A_only_first20":only_a[:20],"B_only_first20":only_b[:20]})
    return rows


def audit_fold(a_root: Path, b_root: Path, c_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str,str]], List[Dict[str,str]]]:
    key = fold_key_from_c_dir(c_dir); model, subject, grid, heldout, rank = key
    ad, bd = a_dir(a_root,key), b_dir(b_root,key)
    row: Dict[str,Any] = {
        "model":model,"subject_id":subject,"grid":grid,"heldout_session":heldout,"M_requested":rank,
        "A_fold_dir":str(ad),"B_fold_dir":str(bd),"C_fold_dir":str(c_dir),
        "A_done":(ad/"DONE.json").exists(),"A_skipped":(ad/"SKIPPED.json").exists(),
        "B_manifest":(bd/"B_to_C_manifest.json").exists(),"C_done":(c_dir/"DONE.json").exists(),
        "C_skipped":(c_dir/"SKIPPED.json").exists(),
    }
    if row["C_skipped"]:
        payload=json.loads((c_dir/"SKIPPED.json").read_text(encoding="utf-8")); row["status"]="C_skipped"
        row["C_skip_status"]=payload.get("status"); row.update({f"rank_{k}":v for k,v in payload.get("rank_policy",{}).items() if k in {"analysis_rank","status","covers_fixed_topk","covers_r99_auxiliary"}})
        return row, [], []
    if not row["B_manifest"]: row["status"]="missing_B"; return row,[],[]
    if not row["C_done"]: row["status"]="missing_C"; return row,[],[]
    bl=canonical_b_ledger(bd/"B_to_C_coordinates.npz"); cl=canonical_csv_ledger(c_dir/"heldout_ledger.csv")
    bc=compare_ledgers(bl,cl); row["B_C_ledger_status"]=bc["status"]; row["B_C_mismatches"]=bc["mismatch_counts"]
    if bc["status"]!="pass": row["status"]="B_C_ledger_failure"
    elif row["A_skipped"]: row["status"]="A_skipped"
    elif not row["A_done"]: row["status"]="A_missing"
    else:
        al=canonical_csv_ledger(ad/"heldout_ledger.csv"); ab=compare_ledgers(al,bl); ac=compare_ledgers(al,cl)
        row.update({"A_B_ledger_status":ab["status"],"A_C_ledger_status":ac["status"],"A_B_mismatches":ab["mismatch_counts"],"A_C_mismatches":ac["mismatch_counts"]})
        row["status"]="pass" if ab["status"]==ac["status"]=="pass" else "A_ledger_warning"
    with (bd/"B_to_C_manifest.json").open("r",encoding="utf-8") as f: bm=json.load(f)
    with (c_dir/"fold_summary.json").open("r",encoding="utf-8") as f: cs=json.load(f)
    row.update({
        "B_retained_rank_r99_auxiliary":bm.get("retained_rank_r99"),
        "B_handoff_rank_fixed_topk_requested":bm.get("handoff_rank_fixed_topk_requested"),
        "B_candidate_rank_fixed_topk_block_complete":bm.get("candidate_rank_fixed_topk_block_complete"),
        "B_handoff_selection_policy":bm.get("handoff_selection_policy"),
        "C_analysis_rank":cs.get("C_analysis_rank"),"rank_ratio":cs.get("rank_ratio"),"C_rank_policy_status":cs.get("C_rank_policy_status"),
    })
    return row, csv_read(c_dir/"A_to_C_semantic_retention.csv"), csv_read(c_dir/"internal_semantic_retention.csv")


def preflight(a_root: Path, b_root: Path, outdir: Path, strict: bool=False) -> Dict[str,Any]:
    manifests=sorted(b_root.glob("*/sub-*/grid-*/heldout-session-*/svd-rank-*/B_to_C_manifest.json")); rows=[]; diffs=[]
    for mp in manifests:
        with mp.open("r",encoding="utf-8") as f: m=json.load(f)
        key=(str(m["model"]),int(m["subject_id"]),str(m["grid"]),int(m["heldout_session"]),int(m["svd_rank_requested"]))
        ad=a_dir(a_root,key); bd=mp.parent
        row={"model":key[0],"subject_id":key[1],"grid":key[2],"heldout_session":key[3],"M_requested":key[4],"A_fold_dir":str(ad),"B_fold_dir":str(bd)}
        if (ad/"SKIPPED.json").exists(): row["status"]="A_skipped"
        elif not (ad/"DONE.json").exists(): row["status"]="A_missing"
        else:
            al=canonical_csv_ledger(ad/"heldout_ledger.csv"); bl=canonical_b_ledger(bd/"B_to_C_coordinates.npz"); cmp=compare_ledgers(al,bl)
            row["status"]="pass" if cmp["status"]=="pass" else "A_B_ledger_warning"; row["mismatch_counts"]=cmp["mismatch_counts"]
            for d in run_level_index_diff(al,bl): diffs.append({**{k:row[k] for k in ("model","subject_id","grid","heldout_session","M_requested")},**d})
        rows.append(row)
    ensure_dir(outdir); csv_write(outdir/"preflight_A_B_ledger.csv",rows); csv_write(outdir/"preflight_A_B_run_differences.csv",diffs)
    hard=[r for r in rows if r["status"] not in {"pass","A_skipped","A_missing","A_B_ledger_warning"}]
    payload={"script_version":SCRIPT_VERSION,"finished_at":now(),"n_manifests":len(manifests),"status_counts":{x:sum(r["status"]==x for r in rows) for x in sorted(set(r["status"] for r in rows))},"n_hard_failures":len(hard)}
    json_dump(outdir/"preflight_summary.json",payload)
    if strict and hard: raise RuntimeError(f"Preflight hard failures: {len(hard)}")
    return payload


def integrate(a_root: Path,b_root: Path,c_root: Path,outdir: Path,strict: bool) -> Dict[str,Any]:
    c_dirs=sorted({p.parent for p in (c_root/"folds").glob("*/*/*/*/*/DONE.json")} | {p.parent for p in (c_root/"folds").glob("*/*/*/*/*/SKIPPED.json")})
    if not c_dirs: raise RuntimeError(f"No completed or skipped C folds under {c_root}")
    audits=[]; external=[]; internal=[]
    for cdir in c_dirs:
        row, ext, inte = audit_fold(a_root,b_root,cdir); audits.append(row); external.extend(ext); internal.extend(inte)
    failures=[r for r in audits if r.get("status") not in TOLERATED_STATUSES]
    ensure_dir(outdir); csv_write(outdir/"integration_fold_audit.csv",audits); csv_write(outdir/"A_to_C_semantic_retention.csv",external); csv_write(outdir/"internal_semantic_retention.csv",internal)
    metrics=[]
    for cdir in c_dirs: metrics.extend(csv_read(cdir/"outer_metrics.csv"))
    metrics=[r for r in metrics if r.get("arm") in ARMS]
    groups: Dict[Tuple[str,...],List[Dict[str,str]]] = {}
    for r in metrics:
        key=(r.get("model",""),r.get("M_requested",""),r.get("axis_source",""),r.get("axis_rank",""),r.get("rank_ratio",""),r.get("scenario",""),r.get("arm",""))
        groups.setdefault(key,[]).append(r)
    summary=[]
    for key,rows in groups.items():
        summary.append({"model":key[0],"M_requested":key[1],"axis_source":key[2],"axis_rank":key[3],"rank_ratio":key[4],"scenario":key[5],"arm":key[6],"n_folds":len(rows),"test_bacc_mean":finite_mean(as_float(r.get("test_balanced_accuracy")) for r in rows),"test_bacc_sem":finite_sem(as_float(r.get("test_balanced_accuracy")) for r in rows)})
    csv_write(outdir/"summary_by_model_rank_source_scenario_arm.csv",summary)
    csv_write(outdir/"headline_nontrivial_slow_prefix.csv",[
        r for r in summary if r.get("axis_source")=="slow_prefix"
        and as_float(r.get("rank_ratio")) < 1.0 - 1e-12
    ])
    ret_groups: Dict[Tuple[str,...],List[Dict[str,str]]] = {}
    for r in internal:
        if r.get("status") != "ok": continue
        key=(r.get("model",""),r.get("M_requested",""),r.get("axis_source",""),r.get("axis_rank",""),r.get("scenario",""),r.get("arm",""))
        ret_groups.setdefault(key,[]).append(r)
    ret_summary=[]
    for key,rows in ret_groups.items():
        source_mean=finite_mean(as_float(r.get("source_test_balanced_accuracy")) for r in rows); ambient_mean=finite_mean(as_float(r.get("ambient_test_balanced_accuracy")) for r in rows); chance=as_float(rows[0].get("chance")); denom=ambient_mean-chance
        ret_summary.append({"model":key[0],"M_requested":key[1],"axis_source":key[2],"axis_rank":key[3],"scenario":key[4],"arm":key[5],"n_folds":len(rows),"source_test_bacc_mean":source_mean,"ambient_test_bacc_mean":ambient_mean,"source_minus_ambient_delta_mean":finite_mean(as_float(r.get("source_minus_ambient_delta")) for r in rows),"aggregate_chance_normalized_retention":(source_mean-chance)/denom if np.isfinite(denom) and abs(denom)>1e-12 else float("nan")})
    csv_write(outdir/"internal_retention_summary.csv",ret_summary)
    status_counts={x:sum(r.get("status")==x for r in audits) for x in sorted(set(str(r.get("status")) for r in audits))}
    payload={"script_version":SCRIPT_VERSION,"finished_at":now(),"A_root":str(a_root),"B_root":str(b_root),"C_root":str(c_root),"n_C_folds":len(c_dirs),"n_passed":sum(r.get("status")=="pass" for r in audits),"n_skipped":sum(r.get("status") in {"C_skipped","A_skipped"} for r in audits),"n_warnings":sum(r.get("status") in {"A_missing","A_ledger_warning"} for r in audits),"n_failed":len(failures),"status_counts":status_counts,"failed_folds":failures}
    json_dump(outdir/"ABC_integration_summary.json",payload)
    if strict and failures: raise RuntimeError(f"ABC integration hard failures={len(failures)} status_counts={status_counts}")
    return payload


def self_test() -> Dict[str,Any]:
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); a=root/"A"; b=root/"B"; c=root/"C"; out=root/"out"
        key=("BIOT",9,"raw",1,10); ad=a_dir(a,key); bd=b_dir(b,key); cd=c/"folds"/"BIOT"/"sub-009"/"grid-raw"/"heldout-session-1"/"svd-rank-10"
        for p in (ad,bd,cd): ensure_dir(p)
        rows=[{"global_index":i,"session_id":1,"run_id":1,"sample_start":i,"sample_end":i+1,"phase_id":int(i>1),"task_id":0,"state_label":0 if i<2 else 1} for i in range(4)]
        csv_write(ad/"heldout_ledger.csv",rows); csv_write(cd/"heldout_ledger.csv",rows); json_dump(ad/"DONE.json",{}); json_dump(cd/"DONE.json",{})
        json_dump(cd/"fold_summary.json",{"C_analysis_rank":2,"rank_ratio":0.2,"C_rank_policy_status":"full_B_candidate"})
        csv_write(cd/"A_to_C_semantic_retention.csv",[]); csv_write(cd/"internal_semantic_retention.csv",[]); csv_write(cd/"outer_metrics.csv",[])
        np.savez_compressed(bd/"B_to_C_coordinates.npz",global_index=np.arange(4),session_id=np.ones(4),run_id=np.ones(4),sample_start=np.arange(4),sample_end=np.arange(4)+1,phase_id=np.array([0,0,1,1]),task_id=np.zeros(4),state_label=np.array([0,0,1,1]),is_heldout_session=np.ones(4))
        json_dump(bd/"B_to_C_manifest.json",{"retained_rank_r99":2,"handoff_rank_fixed_topk_requested":2,"candidate_rank_fixed_topk_block_complete":2,"handoff_selection_policy":"pre_registered_fixed_topk_then_nonchaining_block_completion"})
        result=integrate(a,b,c,out,strict=True)
        if result["n_failed"]!=0: raise AssertionError(result)
        # tolerated skip
        (cd/"DONE.json").unlink(); json_dump(cd/"SKIPPED.json",{"status":"scientific_skip","rank_policy":{}})
        result2=integrate(a,b,c,out,strict=True)
        if result2["n_skipped"]!=1: raise AssertionError(result2)
        return {"status":"passed","completed_and_skip_paths":"passed"}


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Audit and aggregate integrated Experiment A/B/C v5.0 results")
    p.add_argument("--a-root",type=Path); p.add_argument("--b-root",type=Path); p.add_argument("--c-root",type=Path); p.add_argument("--outdir",type=Path)
    p.add_argument("--strict",action="store_true"); p.add_argument("--preflight",action="store_true"); p.add_argument("--self-test",action="store_true")
    return p

def main() -> None:
    args=build_parser().parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2)); return
    for name in ("a_root","b_root","outdir"):
        if getattr(args,name) is None: raise ValueError(f"--{name.replace('_','-')} is required")
    if args.preflight:
        print(json.dumps(preflight(args.a_root,args.b_root,args.outdir,args.strict),indent=2,ensure_ascii=False)); return
    if args.c_root is None: raise ValueError("--c-root is required unless --preflight is used")
    print(json.dumps(integrate(args.a_root,args.b_root,args.c_root,args.outdir,args.strict),indent=2,ensure_ascii=False))

if __name__=="__main__": main()
