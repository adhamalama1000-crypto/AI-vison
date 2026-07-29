"""
Command-line entry point for the industrial component detector workflow.

The end-to-end pipeline, in the order you run it::

    # 1. what data exists, what it covers, what it does not
    python -m training.electrical.cli plan
    python -m training.electrical.cli gap

    # 2. fetch every verified public source and remap it onto the taxonomy
    export ROBOFLOW_API_KEY=...
    python -m training.electrical.cli download --all --dst data/raw

    # 3. merge, re-split 80/10/10 without leaking a capture group, inspect
    python -m training.electrical.cli merge   --roots data/raw/rf_* --dst data/merged
    python -m training.electrical.cli split   --src data/merged --dst data/final
    python -m training.electrical.cli analyse --root data/final
    python -m training.electrical.cli gap     --root data/final

    # 4. train, evaluate, export, install
    python -m training.electrical.cli train   --data data/final/dataset.yaml --arch yolo11s --device 0
    python -m training.electrical.cli eval    --root data/final --backend industrial_ultralytics
    python -m training.electrical.cli export  --weights runs/electrical/yolo11s/weights/best.pt --out dist/model --install

    # closing the long tail: pre-label new captures, correct them, retrain
    python -m training.electrical.cli labelguide
    python -m training.electrical.cli autolabel --images captures/ --out data/prelabelled

Other subcommands::

    python -m training.electrical.cli synth   --out data/synth --crops data/crops
    python -m training.electrical.cli remap   --src raw/rf1 --dst data/rf1 --names-from raw/rf1/data.yaml
    python -m training.electrical.cli bench   --data data/final/dataset.yaml --root data/final
    python -m training.electrical.cli tune    --root data/final --backend industrial_onnx
    python -m training.electrical.cli verify  --bundle dist/model
    python -m training.electrical.cli tensorrt

Every subcommand prints JSON (or a table) and exits non-zero on failure, so it
composes into CI. Nothing here silently substitutes a default: an unavailable
architecture, a dead upstream dataset or a missing API key is reported as
skipped/failed, with the reason and the fix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# Allow `python training/electrical/cli.py` as well as `-m`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rtsp_backend.electrical import metrics as em  # noqa: E402
from rtsp_backend.electrical import taxonomy as tax  # noqa: E402
from training.electrical import autolabel as al  # noqa: E402
from training.electrical import datasets as ds  # noqa: E402
from training.electrical import dedup as dd  # noqa: E402
from training.electrical import download as dl  # noqa: E402
from training.electrical import export as ex  # noqa: E402
from training.electrical import prodeval as pe  # noqa: E402
from training.electrical import refine as rf  # noqa: E402
from training.electrical import split as splitter  # noqa: E402
from training.electrical import synthetic as syn  # noqa: E402
from training.electrical import train as tr  # noqa: E402
from training.electrical import transfer as xf  # noqa: E402


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def bm_defaults() -> tuple[int, int, float]:
    """Benchmark defaults, read from the bench module so they cannot drift."""
    from training.electrical import bench as bm

    return bm.DEFAULT_WARMUP, bm.DEFAULT_RUNS, bm.DEFAULT_LATENCY_BUDGET_MS


def cmd_plan(args) -> int:
    _dump({
        "taxonomy": {"class_count": len(tax.CLASS_ORDER),
                     "classes": list(tax.CLASS_ORDER)},
        "dataset_plan": ds.plan(args.sources or None,
                                include_excluded=args.include_excluded),
        "custom_collection": ds.custom_collection_plan(),
        "ultralytics": dict(zip(("available", "version"),
                                tr.ultralytics_available())),
        "architectures": {a: tr.arch_available(a) for a in tr.SUPPORTED_ARCHS},
    })
    return 0


def cmd_download(args) -> int:
    keys = list(args.sources or [])
    if not keys and not args.all:
        _stderr("error: pass --sources <key> ... or --all. "
                "Run 'plan' to see the available source keys.")
        return 2
    if args.locator and len(keys) != 1:
        _stderr("error: --locator overrides one source's locator, so exactly "
                "one --sources key must be given with it.")
        return 2

    if args.locator:
        res = dl.download_source(keys[0], args.dst, locator=args.locator,
                                 keep_raw=args.keep_raw, log=_stderr)
        _dump(res.to_dict())
        return 0 if res.status == "downloaded" else 1

    manifest = dl.download_all(args.dst, keys or None, keep_raw=args.keep_raw,
                               log=_stderr)
    _dump(manifest)
    _stderr(f"\ndownloaded: {len(manifest['downloaded'])}  "
            f"skipped: {len(manifest['skipped'])}  "
            f"failed: {len(manifest['failed'])}")
    _stderr(manifest["attribution_note"])
    # A batch download with nothing to show for it is a failure, even though each
    # individual source reported its own reason.
    return 0 if manifest["downloaded"] else 1


def cmd_split(args) -> int:
    try:
        report = splitter.split_dataset(
            args.src, args.dst,
            ratios=(args.train, args.val, args.test),
            seed=args.seed, groups_json=args.groups,
            symlink=args.symlink, log=_stderr)
    except (ValueError, OSError) as exc:
        _stderr(f"error: {exc}")
        return 1
    _dump(report)
    return 0


def cmd_scope(args) -> int:
    """List class profiles, or filter a dataset down to one.

    Named cmd_scope, not cmd_profile: cmd_profile is the runtime latency profiler
    further down this file, and defining two functions with one name silently keeps
    only the last.
    """
    from training.electrical import profiles as pf

    if args.list or not args.src:
        out = pf.list_profiles()
        if args.name:
            prof = pf.get(args.name)
            out["selected"] = prof.to_dict()
            out["requirement"] = pf.requirement_estimate(prof, args.target_map)
        _dump(out)
        if not args.src and not args.list:
            _stderr("\n(no --src given, so nothing was filtered — pass --src and "
                    "--dst to produce a profile dataset)")
        return 0

    if not args.dst:
        _stderr("error: --dst is required with --src")
        return 2
    try:
        prof = pf.get(args.name or pf.DEFAULT_PROFILE)
    except KeyError as exc:
        _stderr(f"error: {exc}")
        return 2

    if args.only_present:
        present = pf.present_classes(args.src, prof,
                                     min_instances=args.min_instances)
        if not present:
            _stderr(f"error: none of profile '{prof.name}'s classes have at least "
                    f"{args.min_instances} instance(s) in {args.src}")
            return 1
        if len(present) < prof.class_count:
            _stderr(f"narrowing '{prof.name}' from {prof.class_count} to "
                    f"{len(present)} class(es) with data — an absent class adds a "
                    f"zero to the mAP mean and nothing to the model")
        prof = pf.derive(prof, present)

    stats = pf.apply(args.src, args.dst, prof, drop_empty=args.drop_empty,
                     symlink=args.symlink, log=_stderr)
    stats["profile_detail"] = prof.to_dict()
    stats["requirement"] = pf.requirement_estimate(prof, args.target_map)
    _dump(stats)
    for w in stats["warnings"]:
        _stderr(f"\nwarning: {w}")
    # No usable instances means the filter produced nothing trainable.
    return 0 if stats["instances_kept"] else 1


def cmd_quality(args) -> int:
    """Structural and image-quality inspection of a dataset."""
    from training.electrical import quality as ql

    if args.dst:
        out = ql.clean(args.root, args.dst,
                       drop_warnings=tuple(args.drop_warnings or ()),
                       quarantine=not args.no_quarantine, log=_stderr)
        _dump(out)
        report = out["quality_report"]
    else:
        rep = ql.inspect(args.root, check_pixels=not args.no_pixels,
                         log=_stderr)
        report = rep.to_dict()
        _dump(report)

    _stderr("\n" + report["verdict"])
    for rec in report["recommendations"]:
        _stderr(f"\n- {rec}")
    # Unusable files are a hard failure: training on them wastes the run.
    return 1 if report["fatal_count"] else 0


def cmd_gap(args) -> int:
    """Exactly what is missing: classes, annotations, images, what to collect."""
    analysis = None
    if args.root:
        analysis = ds.analyse_dataset(args.root)
    report = ds.requirements_report(analysis, target=args.target,
                                    priority_only=args.priority_only)
    _dump({"dataset_root": args.root, "analysis": analysis,
           "requirements": report})
    _stderr("\n" + report["summary"])
    if report["missing_classes"]:
        _stderr(f"\nclasses with ZERO annotations "
                f"({len(report['missing_classes'])}): "
                f"{', '.join(report['missing_classes'])}")
    # Non-zero while the priority classes are not ready, so CI can gate a
    # release on "the model has data behind it".
    return 0 if not report["priority"]["missing_classes"] else 1


def cmd_autolabel(args) -> int:
    manifest = al.autolabel_directory(
        args.images, args.out, backends=args.backends or al.DEFAULT_BACKENDS,
        params=json.loads(args.params) if args.params else None,
        accept=args.accept, review=args.review, split=args.split,
        limit=args.limit, copy_images=not args.symlink,
        refine_boxes=not args.no_refine, sam_weights=args.sam_weights,
        device=args.device, log=_stderr)
    _dump(manifest)
    ref = manifest.get("box_refinement") or {}
    if ref.get("interpretation"):
        _stderr("\nbox refinement: " + ref["interpretation"])
    elif not ref.get("enabled") and ref.get("reason"):
        _stderr("\nbox refinement unavailable: " + str(ref["reason"]))
    return 0 if manifest.get("status") == "labelled" else 1


def cmd_labelguide(args) -> int:
    _dump(al.annotation_instructions())
    return 0


def cmd_export(args) -> int:
    evaluation = None
    if args.eval_json:
        try:
            with open(args.eval_json, "r", encoding="utf-8") as fh:
                evaluation = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            _stderr(f"error: could not read --eval-json {args.eval_json}: {exc}")
            return 2

    res = ex.export_bundle(
        args.weights, args.out, imgsz=args.imgsz, opset=args.opset,
        simplify=not args.no_simplify, half=args.half, dynamic=args.dynamic,
        metadata={"data": args.data, "notes": args.notes} if
        (args.data or args.notes) else None,
        run_dir=args.run_dir, evaluation=evaluation, plots=not args.no_plots,
        log=_stderr)
    if res.get("status") != "exported":
        _dump(res)
        return 1
    if args.install:
        res["install"] = ex.install_bundle(args.out, args.install_dir,
                                          log=_stderr)
    _dump(res)
    for w in res.get("warnings") or []:
        _stderr(f"warning: {w}")
    if args.install and res["install"]["status"] != "installed":
        return 1
    return 0 if res["verification"]["ok"] else 1


def cmd_analyse_batch(args) -> int:
    """Run the detector over a folder, batching the forward pass."""
    import time

    import cv2

    from rtsp_backend import electrical  # noqa: F401  (registers backends)
    from rtsp_backend.ai import registry

    if not os.path.isdir(args.images):
        _stderr(f"error: not a directory: {args.images}")
        return 2
    params = json.loads(args.params) if args.params else {}
    try:
        backend = registry.get("components", args.backend)(**params)
        backend.load()
    except Exception as exc:
        _stderr(f"backend unavailable: {exc}")
        return 1
    if not getattr(backend, "ready", False):
        _stderr(f"backend not ready: {getattr(backend, '_reason', 'unknown')}")
        return 1

    files = [f for f in sorted(os.listdir(args.images))
             if f.lower().endswith(ds.IMAGE_EXTS)]
    if args.limit:
        files = files[:args.limit]
    if not files:
        _stderr(f"no images in {args.images}")
        return 1

    frames, names, unreadable = [], [], []
    for fn in files:
        img = cv2.imread(os.path.join(args.images, fn), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(fn)
            continue
        frames.append(img)
        names.append(fn)

    started = time.perf_counter()
    if hasattr(backend, "recognize_batch"):
        results = backend.recognize_batch(frames, batch_size=args.batch)
    else:
        results = [backend.recognize(f) for f in frames]
    elapsed = time.perf_counter() - started

    per_image = []
    from collections import Counter
    totals: Counter = Counter()
    for fn, res in zip(names, results):
        counts = Counter(c.class_id for c in res.accepted)
        totals.update(counts)
        per_image.append({"filename": fn, "components": len(res.accepted),
                          "counts": dict(counts)})

    _dump({
        "backend": args.backend,
        "true_batching": bool(getattr(backend, "supports_true_batching", False)),
        "batch_size": args.batch,
        "images": len(frames),
        "unreadable": unreadable,
        "elapsed_s": round(elapsed, 3),
        "images_per_second": (round(len(frames) / elapsed, 3) if elapsed else None),
        "ms_per_image": (round(elapsed * 1000.0 / len(frames), 2)
                         if frames else None),
        "total_components": int(sum(totals.values())),
        "instances_per_class": dict(totals.most_common()),
        "per_image": per_image,
        "note": (
            "Batching is a throughput optimisation for folder-scale work. It is "
            "deliberately not used on the RTSP path: a cabinet does not change "
            "between frames, so buffering to fill a batch would add latency for no "
            "accuracy gain."
            + ("" if getattr(backend, "supports_true_batching", False) else
               f" NOTE: '{args.backend}' does not implement a real batched forward "
               f"pass, so this ran sequentially — the timing above is per-image "
               f"cost, not a batching speed-up.")),
    })
    return 0


def cmd_hpo(args) -> int:
    from training.electrical import hpo

    space = hpo.HpoSpace(frozen=tuple(args.freeze or ()))
    res = hpo.optimise(
        args.data, dataset_root=args.root, arch=args.arch,
        trials=args.trials, epochs=args.epochs, device=args.device,
        space=space,
        respect_domain_priors=not args.no_respect_domain_priors,
        study_name=args.study, storage=args.storage,
        prune=not args.no_prune, seed=args.seed,
        baseline=not args.no_baseline, log=_stderr)

    if res.status != "completed":
        _dump(res.to_dict())
        _stderr(f"{res.status}: {res.reason}")
        return 1

    _stderr("\n" + hpo.format_history(res))
    _dump(res.to_dict())
    if args.out:
        hpo.write_result(res, args.out)
        _stderr(f"\nwrote {args.out}")
    for w in res.warnings:
        _stderr(f"\nwarning: {w}")
    _stderr("\n" + (res.next_step or ""))
    return 0


def cmd_verify(args) -> int:
    info = ex.verify_bundle(args.bundle)
    _dump(info)
    for p in info["problems"]:
        _stderr(f"problem: {p}")
    return 0 if info["ok"] else 1


def cmd_tensorrt(args) -> int:
    _dump(ex.tensorrt_instructions(args.bundle, args.imgsz))
    return 0


def cmd_synth(args) -> int:
    def progress(split: str, done: int, total: int) -> None:
        print(f"  {split}: {done}/{total}", file=sys.stderr)

    manifest = syn.write_dataset(
        args.out, n_train=args.train, n_val=args.val,
        width=args.width, height=args.height,
        crop_library=args.crops, seed=args.seed, progress=progress)
    _dump(manifest)
    return 0


def cmd_remap(args) -> int:
    if args.names_from:
        names = ds.read_yolo_names(args.names_from)
    elif args.names:
        names = list(args.names)
    else:
        print("error: pass --names-from <data.yaml> or --names a b c",
              file=sys.stderr)
        return 2
    label_map = {}
    if args.source_key and args.source_key in ds.SOURCE_INDEX:
        label_map = dict(ds.SOURCE_INDEX[args.source_key].label_map)
    stats = ds.remap_yolo_dataset(args.src, args.dst, names, label_map,
                                  copy_images=not args.symlink,
                                  prefix=args.prefix or "")
    _dump(stats)
    if stats["unmapped_source_classes"]:
        print(f"warning: {len(stats['unmapped_source_classes'])} source class(es) "
              f"could not be mapped and their instances were dropped: "
              f"{stats['unmapped_source_classes']}", file=sys.stderr)
    return 0


def cmd_merge(args) -> int:
    totals = ds.merge(args.roots, args.dst, copy_images=not args.symlink)
    out = {"merge": totals}
    if args.dedup:
        # Merging is exactly when duplicates appear: two sources republishing the
        # same photographs is the common case, not the exception.
        dedup_dst = args.dedup_dst or (args.dst.rstrip("/\\") + "_dedup")
        out["dedup"] = dd.deduplicate(
            args.dst, dedup_dst, threshold=args.dedup_threshold,
            symlink=args.symlink, log=_stderr)
        out["dataset_root"] = dedup_dst
    else:
        out["dataset_root"] = args.dst
        out["dedup_hint"] = (
            "Duplicates were NOT checked. Two of the registry's public sources are "
            "probably the same photographs republished; if both are in this merge, "
            "the same image can land in train and val and every validation number "
            "will be inflated. Run: python -m training.electrical.cli dedup "
            f"--root {args.dst}")
    _dump(out)
    return 0


def cmd_dedup(args) -> int:
    if args.dst:
        report = dd.deduplicate(args.root, args.dst,
                                threshold=args.threshold, keep=args.keep,
                                symlink=args.symlink,
                                drop_label_conflicts=args.drop_label_conflicts,
                                log=_stderr)
        _dump(report)
        for w in report.get("warnings") or []:
            _stderr(f"warning: {w}")
        return 0

    report = dd.analyse_duplicates(args.root, threshold=args.threshold,
                                   log=_stderr)
    _dump(report)
    if report.get("status") != "analysed":
        _stderr(f"skipped: {report.get('reason')}")
        return 1
    _stderr("\n" + report["verdict"])
    if not args.dst:
        _stderr("\n(read-only analysis — pass --dst to write a deduplicated copy)")
    # Cross-split duplication corrupts every metric, so make it a CI failure.
    return 1 if report["cross_split_groups"] else 0


def cmd_analyse(args) -> int:
    analysis = ds.analyse_dataset(args.root)
    coverage = ds.coverage_report(analysis)
    _dump({"analysis": analysis, "coverage": coverage})
    print("\n" + coverage["summary"], file=sys.stderr)
    return 0


def cmd_train(args) -> int:
    cfg = tr.TrainConfig(
        data=args.data, arch=args.arch, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name)
    res = tr.train(cfg, export_onnx=not args.no_export,
                   log=lambda m: print(m, file=sys.stderr))
    _dump(res.to_dict())
    if res.status == "trained" and res.onnx and args.install:
        dest_dir = os.path.join("models", "components")
        os.makedirs(dest_dir, exist_ok=True)
        import shutil
        dest = os.path.join(dest_dir, os.path.basename(res.onnx))
        shutil.copy2(res.onnx, dest)
        tr._write_classes_json(dest_dir)
        print(f"installed {dest} — the backend will pick it up on next load",
              file=sys.stderr)
    return 0 if res.status == "trained" else 1


def cmd_bench(args) -> int:
    weights = json.loads(args.weights) if args.weights else None
    out = tr.benchmark(args.data, args.root, archs=args.archs,
                       epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       device=args.device,
                       measure_runtime=not args.no_runtime,
                       runs=args.runs,
                       latency_budget_ms=args.latency_budget,
                       weights=weights,
                       min_map_50_95=args.min_map,
                       log=_stderr)
    _stderr("\n=== accuracy ===")
    _stderr(out["table"])
    _stderr("\n=== runtime ===")
    _stderr(out["runtime_table"])
    _stderr("\n=== selection (accuracy + speed) ===")
    _stderr(out["selection_table"])
    _stderr("\n" + out["selection"]["rationale"])
    _dump({k: v for k, v in out.items()
           if k not in ("table", "runtime_table", "selection_table")})
    return 0 if out["selection"]["winner"] else 1


def cmd_profile(args) -> int:
    """Measure latency/FPS/memory of an already-trained checkpoint or backend."""
    from training.electrical import bench as bm

    if args.weights:
        prof = tr.profile_trained(
            args.weights, args.root, args.label or os.path.basename(args.weights),
            split=args.split, imgsz=args.imgsz, device=args.device,
            warmup=args.warmup, runs=args.runs, log=_stderr)
    else:
        from rtsp_backend import electrical  # noqa: F401
        from rtsp_backend.ai import registry
        params = json.loads(args.params) if args.params else {}
        try:
            inst = registry.get("components", args.backend)(**params)
            inst.load()
        except Exception as exc:
            _stderr(f"backend unavailable: {exc}")
            return 1
        prof = bm.profile_backend(
            inst, os.path.join(args.root, "images", args.split),
            args.label or args.backend, warmup=args.warmup, runs=args.runs,
            device=args.device, log=_stderr)

    _dump(prof.to_dict())
    if prof.status != "measured":
        _stderr(f"{prof.status}: {prof.reason}")
        return 1
    _stderr("\n" + bm.format_profile_table({prof.label: prof}))
    for n in prof.notes:
        _stderr(f"note: {n}")
    return 0


def cmd_eval(args) -> int:
    params = json.loads(args.params) if args.params else {}
    rep = tr.evaluate_backend(args.backend, args.root, args.split, params,
                              limit=args.limit)
    if rep.get("status") != "evaluated":
        _stderr(f"skipped: {rep.get('reason')}")
        _dump(rep)
        return 1
    # The table is for a human and goes to stderr; stdout stays pure JSON so that
    # `cli eval ... > eval.json` is machine-readable. It previously went to stdout,
    # which broke exactly the `--eval-json` pipe the export step documents.
    _stderr(em.format_table(em.compare_models({args.backend: rep})))
    _dump(rep)
    return 0


def cmd_tune(args) -> int:
    """Derive per-class confidence thresholds from a validation split."""
    params = json.loads(args.params) if args.params else {}
    gts = tr.load_ground_truth(args.root, args.split)
    if not gts:
        print(f"no ground truth under {args.root}/labels/{args.split}",
              file=sys.stderr)
        return 1
    from rtsp_backend import electrical  # noqa: F401
    from rtsp_backend.ai import registry
    try:
        inst = registry.get("components", args.backend)(**params)
        inst.load()
    except Exception as exc:
        print(f"backend unavailable: {exc}", file=sys.stderr)
        return 1
    preds = tr.collect_predictions(
        inst, os.path.join(args.root, "images", args.split), limit=args.limit)
    rec = em.optimise_thresholds(gts, preds, objective=args.objective,
                                min_precision=args.min_precision)
    _dump({"recommended_thresholds":
           {cid: v["recommended_threshold"] for cid, v in rec.items()},
           "detail": rec,
           "how_to_apply": (
               "Pass these as the 'thresholds' param of the components backend "
               "(POST /api/ai/models/components/params), or edit min_conf in "
               "rtsp_backend/electrical/taxonomy.py to make them the default.")})
    return 0


def _prodeval_cache(args, decode_floor: float):
    """Ground truth + one cached inference pass, or ``(None, None)`` on failure."""
    gts = tr.load_ground_truth(args.root, args.split)
    if not gts:
        _stderr(f"no ground truth under {args.root}/labels/{args.split}")
        return None, None
    params = json.loads(args.params) if args.params else {}
    try:
        cache = pe.cache_candidates(args.backend, args.root, args.split, params,
                                    limit=args.limit,
                                    base_decode_floor=decode_floor,
                                    log=_stderr)
    except Exception as exc:
        _stderr(f"backend unavailable: {exc}")
        return None, None
    if not cache.images:
        _stderr("; ".join(cache.notes) or "no images were evaluated")
        return None, None
    return gts, cache


def cmd_prodeval(args) -> int:
    """Evaluate one operating point through the production inference path."""
    gts, cache = _prodeval_cache(args, args.decode_floor)
    if cache is None:
        return 1
    thresholds = json.loads(args.thresholds) if args.thresholds else None
    rep = pe.production_report(gts, cache, args.decode_floor, args.unknown_floor,
                               thresholds=thresholds, strictness=args.strictness,
                               iou_thr=args.iou)
    _stderr("\n" + pe.format_production(rep["production"]))
    _dump(rep)
    return 0


def cmd_sweep(args) -> int:
    """Sweep decode_floor x unknown_floor and choose a production operating point."""
    decode = sorted(args.decode_floors or list(pe.DECODE_FLOORS))
    unknown = sorted(args.unknown_floors or list(pe.UNKNOWN_FLOORS))
    gts, cache = _prodeval_cache(args, min(decode))
    if cache is None:
        return 1
    _stderr(f"cached {cache.raw_count} raw candidates over {cache.image_count} "
            f"images at decode_floor={cache.base_decode_floor}; replaying "
            f"{len(decode) * len(unknown)} operating points")
    res = pe.sweep(gts, cache, decode, unknown, objective=args.objective,
                   max_fp_per_image=args.max_fp_per_image,
                   min_precision=args.min_precision,
                   min_recall=args.min_recall, iou_thr=args.iou, log=_stderr)
    if res.get("status") != "swept":
        _stderr(res.get("reason", "sweep failed"))
        _dump(res)
        return 1
    if args.per_class:
        best = res["best"]
        ref = pe.refine_per_class(
            gts, cache, best["decode_floor"], best["unknown_floor"],
            objective=args.per_class_objective,
            min_precision=args.min_precision, rank_by=args.objective,
            iou_thr=args.iou, log=_stderr)
        res["per_class_refinement"] = ref
        if ref.get("adopted"):
            res["best"] = ref["tuned"]
            res["best_report"] = ref["report"]
            res["chosen_thresholds"] = ref["thresholds"]
    _stderr("\n" + pe.format_sweep(res, top=args.top))
    _stderr("\n" + pe.format_production(res["best"]))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        _stderr(f"\nwrote {args.out}")
    _dump(res)
    return 0


# --------------------------------------------------------------------------
# synthetic -> real domain transfer
#
# Four subcommands, named `mix` / `gap` is taken / `finetune` / `transfer`. The
# function names are prefixed cmd_xf_* rather than reusing cmd_transfer-style names,
# because a duplicate `def cmd_x` later in this module silently shadows the earlier one
# and the failure surfaces as a missing argparse attribute, not as an import error.
# --------------------------------------------------------------------------

def cmd_xf_mix(args) -> int:
    """Build a real+synthetic training set with a REAL-ONLY validation split."""
    try:
        rep = xf.build_mixed(args.real, args.synth, args.dst,
                             synth_fraction=args.synth_fraction,
                             seed=args.seed, symlink=not args.copy,
                             log=_stderr)
    except ValueError as exc:
        _stderr(f"failed: {exc}")
        _dump({"status": "failed", "reason": str(exc)})
        return 1
    rep["status"] = "built"
    _dump(rep)
    return 0


def cmd_xf_domaingap(args) -> int:
    """Score one checkpoint on synthetic and on real data, and name the gap."""
    rep = xf.measure_domain_gap(args.weights, args.synth, args.real,
                                split=args.split, imgsz=args.imgsz,
                                device=args.device, log=_stderr)
    if args.out:
        _stderr(f"wrote {xf.write_result(rep, args.out)}")
    _dump(rep)
    return 0 if rep.get("real", {}).get("status") == "evaluated" else 1


def cmd_xf_finetune(args) -> int:
    """Fine-tune a checkpoint onto a new domain, optionally in two stages."""
    rep = xf.fine_tune(args.data, args.init_from, arch=args.arch,
                       epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       device=args.device, staged=not args.no_staged,
                       freeze_layers=args.freeze,
                       stage1_epochs=args.stage1_epochs,
                       lr0=args.lr0, name=args.name, log=_stderr)
    if args.out:
        _stderr(f"wrote {xf.write_result(rep, args.out)}")
    _dump(rep)
    return 0 if rep.get("status") == "completed" else 1


def cmd_xf_compare(args) -> int:
    """Train every transfer strategy and rank them on REAL validation data."""
    rep = xf.compare_strategies(
        args.real, args.synth, args.work_dir, strategies=tuple(args.strategies),
        arch=args.arch, epochs=args.epochs,
        synth_pretrain_epochs=args.synth_pretrain_epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        synth_fraction=args.synth_fraction, synth_weights=args.synth_weights,
        log=_stderr)
    # Table to stderr, JSON to stdout, so `> comparison.json` stays parseable.
    _stderr(xf.format_ranking(rep))
    if rep.get("rationale"):
        _stderr("\n" + rep["rationale"])
    if args.out:
        _stderr(f"\nwrote {xf.write_result(rep, args.out)}")
    _dump(rep)
    return 0 if rep.get("status") == "completed" else 1


def cmd_xf_expand(args) -> int:
    """Describe (or run) a staged class expansion, e.g. core8 -> core15."""
    from training.electrical import profiles as pf

    try:
        plan = pf.expansion_plan(args.frm, args.to)
    except KeyError as exc:
        _stderr(str(exc))
        return 1

    _stderr(f"{args.frm} -> {args.to}: "
            f"{plan['from_classes']} -> {plan['to_classes']} classes")
    if plan["added"]:
        _stderr(f"  adds: {', '.join(plan['added'])}")
    if plan["removed"]:
        _stderr(f"  drops: {', '.join(plan['removed'])}")
    _stderr(f"  {plan['guidance']}")

    if not args.data:
        _dump(plan)
        return 0

    if not plan["index_stable"] and args.init_from:
        # Refusing rather than warning. Carrying a checkpoint across a non-prefix
        # profile change produces a model that loads without complaint and predicts
        # the wrong labels — the failure would surface as bad inspection reports.
        _stderr("refusing to fine-tune across a non-prefix profile change; "
                "drop --init-from to train a fresh head instead")
        plan["status"] = "refused"
        _dump(plan)
        return 1

    rep = xf.fine_tune(args.data, args.init_from, arch=args.arch,
                       epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       device=args.device, staged=not args.no_staged,
                       name=f"expand_{args.frm}_to_{args.to}", log=_stderr)
    plan["training"] = rep
    plan["status"] = rep.get("status")
    if args.out:
        _stderr(f"wrote {xf.write_result(plan, args.out)}")
    _dump(plan)
    return 0 if rep.get("status") == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="training.electrical.cli",
        description="Industrial component detector: data, training, evaluation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("plan", help="show taxonomy, dataset sources, capabilities")
    ap.add_argument("--sources", nargs="*", help="restrict to these source keys")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also forecast sources the registry excludes "
                         "(thermal imagery, wrong-domain sets)")
    ap.set_defaults(func=cmd_plan)

    ap = sub.add_parser("download",
                        help="fetch public datasets and remap onto the taxonomy")
    ap.add_argument("--sources", nargs="*", help="source keys (see 'plan')")
    ap.add_argument("--all", action="store_true",
                    help="every downloadable, non-excluded source")
    ap.add_argument("--dst", default="data/raw")
    ap.add_argument("--locator",
                    help="override the registry locator for a single source, "
                         "e.g. your fork: my-workspace/control-panels/1")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the unprocessed download for inspection")
    ap.set_defaults(func=cmd_download)

    ap = sub.add_parser("split",
                        help="re-split 80/10/10 grouped by capture, no leakage")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--groups",
                    help="groups.json mapping image filename -> panel id. "
                         "Beats the filename heuristic; record it at capture "
                         "time.")
    ap.add_argument("--symlink", action="store_true")
    ap.set_defaults(func=cmd_split)

    # Named 'scope' rather than 'profile': 'profile' is already the runtime
    # latency/memory profiler, and that is the standard meaning of the word for a
    # model. This one narrows the *class scope* of training.
    ap = sub.add_parser("scope",
                        help="list class profiles, or filter a dataset to one "
                             "(fewer classes = more instances each = higher mAP)")
    ap.add_argument("--list", action="store_true", help="list profiles and exit")
    ap.add_argument("--name", help="profile name (default: core15)")
    ap.add_argument("--src", help="canonically-labelled dataset root to filter")
    ap.add_argument("--dst", help="output root for the profile dataset")
    ap.add_argument("--drop-empty", action="store_true",
                    help="also drop images left with no in-profile boxes. OFF by "
                         "default: such an image is a genuine NEGATIVE for this "
                         "profile, and negatives teach the detector not to fire on "
                         "out-of-profile devices")
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--only-present", action="store_true",
                    help="narrow the profile to the classes that actually have data "
                         "in --src. An absent class adds a zero to the mAP mean and "
                         "nothing to the model, so reporting a 15-class mAP with 7 "
                         "empty classes misleads in both directions")
    ap.add_argument("--min-instances", type=int, default=1,
                    help="with --only-present, the instance count a class needs to "
                         "be kept")
    ap.add_argument("--target-map", type=float, default=0.85,
                    help="target mAP50 for the data-requirement estimate")
    ap.set_defaults(func=cmd_scope)

    ap = sub.add_parser("quality",
                        help="corrupted files, bad labels, low-quality images, "
                             "class balance; exits non-zero on unusable files")
    ap.add_argument("--root", required=True)
    ap.add_argument("--dst", help="write a cleaned dataset here (rejects go to "
                                  "quarantine/, nothing is deleted)")
    ap.add_argument("--drop-warnings", nargs="*",
                    help="also drop files with these warning codes, e.g. blurred "
                         "too_dark. OFF by default: a dim panel photograph is real "
                         "deployment input, and filtering on image statistics throws "
                         "away the hardest training examples")
    ap.add_argument("--no-quarantine", action="store_true",
                    help="do not keep copies of rejected files")
    ap.add_argument("--no-pixels", action="store_true",
                    help="structural/label checks only; skip image decoding")
    ap.set_defaults(func=cmd_quality)

    ap = sub.add_parser("gap",
                        help="what is missing: classes, annotations, images")
    ap.add_argument("--root", help="dataset root; omit to cost it from zero")
    ap.add_argument("--target", type=int, default=ds.MIN_INSTANCES_RELIABLE,
                    help="target annotations per class")
    ap.add_argument("--priority-only", action="store_true",
                    help="restrict to the priority class list")
    ap.set_defaults(func=cmd_gap)

    ap = sub.add_parser("autolabel",
                        help="pre-label images for human correction")
    ap.add_argument("--images", required=True, help="directory of images")
    ap.add_argument("--out", required=True, help="output YOLO dataset root")
    ap.add_argument("--backends", nargs="*",
                    help=f"backends to try, in order "
                         f"(default: {' '.join(al.DEFAULT_BACKENDS)})")
    ap.add_argument("--params", help="JSON params for the backend")
    ap.add_argument("--accept", type=float, default=al.DEFAULT_ACCEPT)
    ap.add_argument("--review", type=float, default=al.DEFAULT_REVIEW)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip SAM2 box refinement. Open-vocabulary boxes are "
                         "loose (they include DIN rail and neighbouring devices), "
                         "so refinement is on by default — but if the manifest "
                         "reports a low accept rate it is costing compute for "
                         "nothing on your imagery")
    ap.add_argument("--sam-weights",
                    help="explicit SAM/SAM2 checkpoint or HF model id "
                         f"(default: first available of {', '.join(rf.SAM_CANDIDATES[:3])}...)")
    ap.add_argument("--device", default="cpu",
                    help="device for SAM refinement, e.g. 0 for the first GPU")
    ap.set_defaults(func=cmd_autolabel)

    ap = sub.add_parser("labelguide",
                        help="print the human annotation instructions")
    ap.set_defaults(func=cmd_labelguide)

    ap = sub.add_parser("export",
                        help="build best.pt + best.onnx + labels.txt bundle")
    ap.add_argument("--weights", required=True, help="trained best.pt")
    ap.add_argument("--out", required=True, help="bundle output directory")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--half", action="store_true",
                    help="FP16 ONNX — GPU only, re-measure mAP after")
    ap.add_argument("--dynamic", action="store_true",
                    help="dynamic input shape (slower, more flexible)")
    ap.add_argument("--data",
                    help="dataset.yaml the model was trained on. Recorded in the "
                         "model card, and REQUIRED for a profile-trained model: the "
                         "label space is read from it, otherwise the bundle defaults "
                         "to the full 54-class taxonomy and its labels disagree with "
                         "an N-class graph")
    ap.add_argument("--notes", help="free-text provenance for the model card")
    ap.add_argument("--run-dir",
                    help="Ultralytics run directory to harvest curves and the "
                         "confusion matrix from (default: inferred from the "
                         "weights path)")
    ap.add_argument("--eval-json",
                    help="JSON from 'cli eval', recorded into metrics.json. "
                         "Without it the bundle carries no accuracy evidence and "
                         "cannot be audited later.")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip rendering curves/confusion matrix for anything "
                         "Ultralytics did not already plot")
    ap.add_argument("--install", action="store_true",
                    help="install into models/components/ after verification")
    ap.add_argument("--install-dir", default=ex.DEFAULT_INSTALL_DIR)
    ap.set_defaults(func=cmd_export)

    ap = sub.add_parser("analyse-batch",
                        help="run the detector over a folder with batched inference")
    ap.add_argument("--images", required=True)
    ap.add_argument("--backend", default="industrial_onnx")
    ap.add_argument("--params", help="JSON params for the backend")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.set_defaults(func=cmd_analyse_batch)

    ap = sub.add_parser("hpo",
                        help="Optuna hyperparameter search for the detector")
    ap.add_argument("--data", required=True, help="dataset.yaml")
    ap.add_argument("--root", help="dataset root, so the search can warn when "
                                  "the dataset is too thin to be worth tuning")
    ap.add_argument("--arch", default=tr.DEFAULT_ARCH, choices=tr.SUPPORTED_ARCHS)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20,
                   help="epochs PER TRIAL. Short is fine — the search ranks the "
                        "space, then you train the winner properly. Below ~10 the "
                        "ranking stops transferring to a full run.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--study", default="electrical_detector_hpo")
    ap.add_argument("--storage",
                   help="Optuna storage URL (default: sqlite under "
                        "runs/electrical/, so an interrupted search resumes)")
    ap.add_argument("--freeze", nargs="*",
                   help="hyperparameters to leave at their defaults")
    ap.add_argument("--no-prune", action="store_true",
                   help="disable median pruning of trials that fall behind")
    ap.add_argument("--no-baseline", action="store_true",
                   help="skip the hand-tuned reference run")
    ap.add_argument("--no-respect-domain-priors", action="store_true",
                   help="also search fliplr/flipud and unbounded rotation. OFF by "
                        "default: the search will turn horizontal flip on because "
                        "it looks like free augmentation, and teach the model that "
                        "mirrored nameplates are normal")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", help="write the full result JSON here")
    ap.set_defaults(func=cmd_hpo)

    ap = sub.add_parser("verify", help="check a bundle the way the runtime will")
    ap.add_argument("--bundle", default=ex.DEFAULT_INSTALL_DIR)
    ap.set_defaults(func=cmd_verify)

    ap = sub.add_parser("tensorrt", help="TensorRT build instructions")
    ap.add_argument("--bundle", default=ex.DEFAULT_INSTALL_DIR)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.set_defaults(func=cmd_tensorrt)

    sp = sub.add_parser("synth", help="generate a labelled synthetic dataset")
    sp.add_argument("--out", required=True)
    sp.add_argument("--train", type=int, default=400)
    sp.add_argument("--val", type=int, default=80)
    sp.add_argument("--width", type=int, default=1024)
    sp.add_argument("--height", type=int, default=768)
    sp.add_argument("--crops", help="crop library root (one dir per class) — "
                                    "strongly recommended; real crops make the "
                                    "output usable for real-world training")
    sp.add_argument("--seed", type=int, default=1234)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("remap", help="remap a YOLO dataset onto the taxonomy")
    sp.add_argument("--src", required=True)
    sp.add_argument("--dst", required=True)
    sp.add_argument("--names-from", help="path to the source data.yaml")
    sp.add_argument("--names", nargs="*", help="source class names, in order")
    sp.add_argument("--source-key", help="registry key for its label map")
    sp.add_argument("--prefix", help="prefix output file names")
    sp.add_argument("--symlink", action="store_true",
                    help="symlink images instead of copying")
    sp.set_defaults(func=cmd_remap)

    sp = sub.add_parser("merge", help="merge remapped datasets")
    sp.add_argument("--roots", nargs="+", required=True)
    sp.add_argument("--dst", required=True)
    sp.add_argument("--symlink", action="store_true")
    sp.add_argument("--dedup", action="store_true",
                    help="remove near-duplicate images after merging "
                         "(strongly recommended — see 'dedup')")
    sp.add_argument("--dedup-dst", help="output root for the deduplicated copy "
                                        "(default: <dst>_dedup)")
    sp.add_argument("--dedup-threshold", type=int, default=dd.DEFAULT_THRESHOLD,
                    help="perceptual-hash Hamming distance threshold")
    sp.set_defaults(func=cmd_merge)

    sp = sub.add_parser("dedup",
                        help="find/remove duplicate images (read-only without "
                             "--dst); exits non-zero on cross-split leakage")
    sp.add_argument("--root", required=True)
    sp.add_argument("--dst", help="write a deduplicated copy here")
    sp.add_argument("--threshold", type=int, default=dd.DEFAULT_THRESHOLD)
    sp.add_argument("--keep", default="train_first",
                    choices=["train_first", "first"],
                    help="train_first keeps the training copy and drops the "
                         "val/test copy, so evaluation stays unseen")
    sp.add_argument("--drop-label-conflicts", action="store_true",
                    help="also deduplicate groups whose labels disagree "
                         "(default: keep them and report, since one label is wrong)")
    sp.add_argument("--symlink", action="store_true")
    sp.set_defaults(func=cmd_dedup)

    sp = sub.add_parser("analyse", help="per-class counts + trainability report")
    sp.add_argument("--root", required=True)
    sp.set_defaults(func=cmd_analyse)

    sp = sub.add_parser("train", help="train one architecture")
    sp.add_argument("--data", required=True, help="dataset.yaml")
    sp.add_argument("--arch", default=tr.DEFAULT_ARCH, choices=tr.SUPPORTED_ARCHS)
    sp.add_argument("--epochs", type=int, default=120)
    sp.add_argument("--imgsz", type=int, default=960)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--name")
    sp.add_argument("--no-export", action="store_true")
    sp.add_argument("--install", action="store_true",
                    help="copy the exported ONNX into models/components/")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("bench",
                        help="train several architectures and pick one on "
                             "accuracy AND speed")
    sp.add_argument("--data", required=True)
    sp.add_argument("--root", required=True, help="dataset root (for evaluation)")
    sp.add_argument("--archs", nargs="+",
                    default=["yolo11s", "yolov8s", "rtdetr-l"])
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--imgsz", type=int, default=960)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--no-runtime", action="store_true",
                    help="skip latency/memory measurement and rank on accuracy "
                         "alone (which reliably picks the biggest model)")
    sp.add_argument("--runs", type=int,
                    help=f"measured inferences per model "
                         f"(default {bm_defaults()[1]})")
    sp.add_argument("--latency-budget", type=float,
                    help=f"disqualify models slower than this p95, ms "
                         f"(default {bm_defaults()[2]:.0f})")
    sp.add_argument("--min-map", type=float, default=0.0,
                    help="disqualify models below this mAP@50-95")
    sp.add_argument("--weights",
                    help='JSON score weights, e.g. '
                         '\'{"map_50_95":0.7,"speed":0.05}\' to prioritise accuracy')
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("profile",
                        help="measure latency / FPS / memory of one model")
    sp.add_argument("--root", required=True,
                    help="dataset root — REAL images are required, because "
                         "detector latency depends on the candidate count")
    sp.add_argument("--split", default="val")
    sp.add_argument("--weights", help="a trained .pt to profile")
    sp.add_argument("--backend", default="industrial_onnx",
                    help="or profile a registered backend instead")
    sp.add_argument("--params", help="JSON params for --backend")
    sp.add_argument("--label")
    sp.add_argument("--imgsz", type=int, default=960)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--warmup", type=int, default=bm_defaults()[0])
    sp.add_argument("--runs", type=int, default=bm_defaults()[1])
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("eval", help="evaluate a registered backend")
    sp.add_argument("--root", required=True)
    sp.add_argument("--backend", default="industrial_onnx")
    sp.add_argument("--split", default="val")
    sp.add_argument("--params", help="JSON params for the backend")
    sp.add_argument("--limit", type=int)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("tune", help="derive per-class confidence thresholds")
    sp.add_argument("--root", required=True)
    sp.add_argument("--backend", default="industrial_onnx")
    sp.add_argument("--split", default="val")
    sp.add_argument("--params")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--objective", default="f1", choices=["f1", "precision", "recall"])
    sp.add_argument("--min-precision", type=float, default=0.0)
    sp.set_defaults(func=cmd_tune)

    # ---- production-path evaluation ----
    sp = sub.add_parser(
        "prodeval",
        help="evaluate ONE operating point through the production inference "
             "path (not Ultralytics val): P/R/mAP plus FP per image, FN per "
             "image, unknown rate, accepted/rejected counts")
    sp.add_argument("--root", required=True)
    sp.add_argument("--backend", default="industrial_onnx")
    sp.add_argument("--split", default="val")
    sp.add_argument("--params", help="JSON params for the backend")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--decode-floor", type=float, default=0.05,
                    help="raw score cutoff applied inside the detector decode")
    sp.add_argument("--unknown-floor", type=float, default=0.18,
                    help="below the per-class threshold but above this, a box is "
                         "kept as 'unknown industrial component' rather than guessed")
    sp.add_argument("--thresholds", help="JSON per-class threshold overrides")
    sp.add_argument("--strictness", type=float, default=1.0)
    sp.add_argument("--iou", type=float, default=em.DEFAULT_IOU)
    sp.set_defaults(func=cmd_prodeval)

    sp = sub.add_parser(
        "sweep",
        help="acceptance sweep: replay the gate over decode_floor x "
             "unknown_floor and pick the best PRODUCTION operating point")
    sp.add_argument("--root", required=True)
    sp.add_argument("--backend", default="industrial_onnx")
    sp.add_argument("--split", default="val")
    sp.add_argument("--params", help="JSON params for the backend")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--decode-floors", type=float, nargs="+",
                    help=f"default {list(pe.DECODE_FLOORS)}")
    sp.add_argument("--unknown-floors", type=float, nargs="+",
                    help=f"default {list(pe.UNKNOWN_FLOORS)}")
    sp.add_argument("--objective", default="production_score",
                    choices=list(pe.OBJECTIVES),
                    help="what to maximise; production_score blends F1 and "
                         "mAP@0.5 and penalises false positives per image")
    sp.add_argument("--max-fp-per-image", type=float,
                    help="reject any operating point noisier than this")
    sp.add_argument("--min-precision", type=float, default=0.0)
    sp.add_argument("--min-recall", type=float, default=0.0)
    sp.add_argument("--per-class", action="store_true",
                    help="also derive per-class thresholds at the winning point, "
                         "and adopt them only if they improve the objective")
    sp.add_argument("--per-class-objective", default="f1",
                    choices=["f1", "precision", "recall"])
    sp.add_argument("--top", type=int, help="only print the N best rows")
    sp.add_argument("--iou", type=float, default=em.DEFAULT_IOU)
    sp.add_argument("--out", help="write the full sweep JSON here")
    sp.set_defaults(func=cmd_sweep)

    # ---- synthetic -> real domain transfer ----
    sp = sub.add_parser(
        "mix", help="build a real+synthetic train split with REAL-ONLY validation")
    sp.add_argument("--real", required=True,
                    help="root of the real dataset (authoritative label space)")
    sp.add_argument("--synth", required=True, help="root of the synthetic dataset")
    sp.add_argument("--dst", required=True)
    sp.add_argument("--synth-fraction", type=float,
                    default=xf.DEFAULT_SYNTH_FRACTION,
                    help="share of the TRAIN split allowed to be synthetic "
                         f"(default {xf.DEFAULT_SYNTH_FRACTION}); val/test are real "
                         "only regardless")
    sp.add_argument("--seed", type=int, default=1234)
    sp.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking")
    sp.set_defaults(func=cmd_xf_mix)

    sp = sub.add_parser(
        "domain-gap",
        help="score a checkpoint on synthetic vs real data and name the gap")
    sp.add_argument("--weights", required=True)
    sp.add_argument("--synth", required=True)
    sp.add_argument("--real", required=True)
    sp.add_argument("--split", default="val")
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--out", help="also write the report to this path")
    sp.set_defaults(func=cmd_xf_domaingap)

    sp = sub.add_parser("finetune",
                        help="fine-tune a checkpoint onto a new domain (staged)")
    sp.add_argument("--data", required=True, help="dataset.yaml (real-validated)")
    sp.add_argument("--init-from",
                    help="checkpoint to start from; omit for COCO pretrained")
    sp.add_argument("--arch", default="yolo11s")
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--no-staged", action="store_true",
                    help="train end-to-end instead of freeze-then-unfreeze")
    sp.add_argument("--freeze", type=int, default=xf.BACKBONE_FREEZE_LAYERS,
                    help="backbone layers to freeze in stage 1 "
                         f"(default {xf.BACKBONE_FREEZE_LAYERS})")
    sp.add_argument("--stage1-epochs", type=int,
                    help="default is a third of --epochs")
    sp.add_argument("--lr0", type=float, default=0.002,
                    help="fine-tuning learning rate; stage 2 uses half of it")
    sp.add_argument("--name", default="finetune")
    sp.add_argument("--out", help="also write the report to this path")
    sp.set_defaults(func=cmd_xf_finetune)

    sp = sub.add_parser(
        "transfer",
        help="train every transfer strategy and rank them on REAL validation")
    sp.add_argument("--real", required=True)
    sp.add_argument("--synth", required=True)
    sp.add_argument("--work-dir", default="runs/electrical/transfer")
    sp.add_argument("--strategies", nargs="+",
                    default=["real_only", "coco_to_synth_to_real", "mixed"],
                    choices=sorted(xf.STRATEGIES))
    sp.add_argument("--arch", default="yolo11s")
    sp.add_argument("--epochs", type=int, default=40)
    sp.add_argument("--synth-pretrain-epochs", type=int, default=20)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--synth-fraction", type=float,
                    default=xf.DEFAULT_SYNTH_FRACTION)
    sp.add_argument("--synth-weights",
                    help="reuse an existing synthetic checkpoint instead of "
                         "pretraining one for coco_to_synth_to_real")
    sp.add_argument("--out", help="also write the comparison to this path")
    sp.set_defaults(func=cmd_xf_compare)

    sp = sub.add_parser(
        "expand",
        help="plan (or run) a staged class expansion, e.g. core8 -> core15")
    sp.add_argument("--from", dest="frm", default="core8")
    sp.add_argument("--to", default="core15")
    sp.add_argument("--data",
                    help="dataset.yaml scoped to the TARGET profile; omit to only "
                         "print the plan")
    sp.add_argument("--init-from",
                    help="checkpoint from the source profile; refused when the "
                         "expansion is not index-stable")
    sp.add_argument("--arch", default="yolo11s")
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--batch", type=int, default=16)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--no-staged", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_xf_expand)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
