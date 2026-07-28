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
from training.electrical import download as dl  # noqa: E402
from training.electrical import export as ex  # noqa: E402
from training.electrical import split as splitter  # noqa: E402
from training.electrical import synthetic as syn  # noqa: E402
from training.electrical import train as tr  # noqa: E402


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


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
        limit=args.limit, copy_images=not args.symlink, log=_stderr)
    _dump(manifest)
    return 0 if manifest.get("status") == "labelled" else 1


def cmd_labelguide(args) -> int:
    _dump(al.annotation_instructions())
    return 0


def cmd_export(args) -> int:
    res = ex.export_bundle(
        args.weights, args.out, imgsz=args.imgsz, opset=args.opset,
        simplify=not args.no_simplify, half=args.half, dynamic=args.dynamic,
        metadata={"data": args.data, "notes": args.notes} if
        (args.data or args.notes) else None,
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
    _dump(ds.merge(args.roots, args.dst, copy_images=not args.symlink))
    return 0


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
    out = tr.benchmark(args.data, args.root, archs=args.archs,
                       epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       device=args.device,
                       log=lambda m: print(m, file=sys.stderr))
    print(out["table"])
    _dump({k: v for k, v in out.items() if k != "table"})
    return 0 if out["comparison"]["winner"] else 1


def cmd_eval(args) -> int:
    params = json.loads(args.params) if args.params else {}
    rep = tr.evaluate_backend(args.backend, args.root, args.split, params,
                              limit=args.limit)
    if rep.get("status") != "evaluated":
        print(f"skipped: {rep.get('reason')}", file=sys.stderr)
        _dump(rep)
        return 1
    print(em.format_table(em.compare_models({args.backend: rep})))
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
    ap.add_argument("--data", help="dataset.yaml used, recorded in the model card")
    ap.add_argument("--notes", help="free-text provenance for the model card")
    ap.add_argument("--install", action="store_true",
                    help="install into models/components/ after verification")
    ap.add_argument("--install-dir", default=ex.DEFAULT_INSTALL_DIR)
    ap.set_defaults(func=cmd_export)

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
    sp.set_defaults(func=cmd_merge)

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

    sp = sub.add_parser("bench", help="train + rank several architectures")
    sp.add_argument("--data", required=True)
    sp.add_argument("--root", required=True, help="dataset root (for evaluation)")
    sp.add_argument("--archs", nargs="+",
                    default=["yolo11s", "yolov8s", "rtdetr-l"])
    sp.add_argument("--epochs", type=int, default=60)
    sp.add_argument("--imgsz", type=int, default=960)
    sp.add_argument("--batch", type=int, default=8)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_bench)

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
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
