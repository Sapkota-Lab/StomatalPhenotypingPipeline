#!/usr/bin/env python3
"""Browser annotator for validated stomatal-complex polygon masks."""

from __future__ import annotations

import argparse
import csv
import getpass
import io
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps


def validate_polygon(
    value: object,
    width: int,
    height: int,
) -> list[list[float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("polygon needs at least three points")
    output = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each polygon point must be [x, y]")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("polygon coordinates must be numeric") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("polygon coordinates must be finite")
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("polygon point is outside the oriented image")
        output.append([round(x, 2), round(y, 2)])
    return output


def polygon_area(points: list[list[float]]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2


def balanced_queue(
    manifest_rows: list[dict[str, str]],
    annotation_dir: Path,
    max_per_image: int,
    seed: int,
) -> list[dict]:
    manifest_by_image = {
        row["stomata_image"]: row for row in manifest_rows
    }
    queues_by_image = []
    for filename, manifest in sorted(manifest_by_image.items()):
        path = annotation_dir / f"{Path(filename).stem}.json"
        if not path.is_file():
            raise ValueError(f"Missing expert boxes for {filename}")
        annotation = json.loads(path.read_text())
        if annotation.get("status") != "complete":
            raise ValueError(f"Expert boxes are not complete for {filename}")
        boxes = list(annotation.get("boxes", []))
        rng = random.Random(f"{seed}:{filename}")
        selected = list(range(len(boxes)))
        rng.shuffle(selected)
        selected = selected[:max_per_image]
        image_queue = []
        for box_index in selected:
            box = boxes[box_index]
            image_queue.append({
                "object_id": f"{Path(filename).stem}:{box_index + 1:04d}",
                "image": filename,
                "expert_box_index": box_index + 1,
                "box": {
                    key: float(box[key])
                    for key in ("x1", "y1", "x2", "y2")
                },
                "plant_unit_id": manifest["plant_unit_id"],
                "genotype_id": manifest["genotype_id"],
                "variety": manifest["variety"],
                "water_status": manifest["water_status"],
            })
        queues_by_image.append(image_queue)
    return [
        image_queue[round_index]
        for round_index in range(
            max((len(items) for items in queues_by_image), default=0)
        )
        for image_queue in queues_by_image
        if round_index < len(image_queue)
    ]


def local_quality_metrics(
    image: Image.Image,
    box: dict,
    context_fraction: float = 0.35,
) -> dict[str, float]:
    """Score local visibility without using object area or mask geometry."""
    width, height = image.size
    box_width = float(box["x2"]) - float(box["x1"])
    box_height = float(box["y2"]) - float(box["y1"])
    left = max(0, math.floor(float(box["x1"]) - context_fraction * box_width))
    top = max(0, math.floor(float(box["y1"]) - context_fraction * box_height))
    right = min(
        width,
        math.ceil(float(box["x2"]) + context_fraction * box_width),
    )
    bottom = min(
        height,
        math.ceil(float(box["y2"]) + context_fraction * box_height),
    )
    patch = image.crop((left, top, right, bottom)).resize(
        (160, 160),
        Image.Resampling.BILINEAR,
    )
    gray = np.asarray(patch.convert("L"), dtype=np.float32)
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * center
    )
    focus = float(np.var(laplacian))
    low, high = np.percentile(gray, [5, 95])
    contrast = float(high - low)
    clipped = float(np.mean((gray <= 3) | (gray >= 252)))
    edge_margin = float(min(
        float(box["x1"]),
        float(box["y1"]),
        width - float(box["x2"]),
        height - float(box["y2"]),
    ))
    # The score deliberately excludes object/mask area and apparent shape.
    score = math.log1p(focus) + 0.025 * contrast - 3.0 * clipped
    if edge_margin < max(box_width, box_height) * 0.25:
        score -= 1.0
    return {
        "local_quality_score": round(score, 6),
        "local_focus_score": round(focus, 6),
        "local_contrast_5_95": round(contrast, 6),
        "local_clipped_fraction": round(clipped, 6),
        "edge_margin_px": round(edge_margin, 2),
    }


def quality_queue(
    manifest_rows: list[dict[str, str]],
    annotation_dir: Path,
    image_dir: Path,
    max_per_image: int,
) -> list[dict]:
    """Select a fixed number of the clearest expert-boxed objects per image."""
    manifest_by_image = {
        row["stomata_image"]: row for row in manifest_rows
    }
    queues_by_image = []
    for filename, manifest in sorted(manifest_by_image.items()):
        annotation_path = annotation_dir / f"{Path(filename).stem}.json"
        if not annotation_path.is_file():
            raise ValueError(f"Missing expert boxes for {filename}")
        annotation = json.loads(annotation_path.read_text())
        if annotation.get("status") != "complete":
            raise ValueError(f"Expert boxes are not complete for {filename}")
        image_path = image_dir / filename
        if not image_path.is_file():
            raise ValueError(f"Missing source image {filename}")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            candidates = []
            for box_index, raw_box in enumerate(annotation.get("boxes", [])):
                box = {
                    key: float(raw_box[key])
                    for key in ("x1", "y1", "x2", "y2")
                }
                candidates.append({
                    "object_id": (
                        f"{Path(filename).stem}:{box_index + 1:04d}"
                    ),
                    "image": filename,
                    "expert_box_index": box_index + 1,
                    "box": box,
                    "plant_unit_id": manifest["plant_unit_id"],
                    "genotype_id": manifest["genotype_id"],
                    "variety": manifest["variety"],
                    "water_status": manifest["water_status"],
                    "selection_method": "local_image_quality_v1",
                    **local_quality_metrics(image, box),
                })
        candidates.sort(
            key=lambda item: (
                -item["local_quality_score"],
                item["expert_box_index"],
            )
        )
        selected = candidates[:max_per_image]
        for rank, item in enumerate(selected, 1):
            item["quality_rank_within_image"] = rank
            item["quality_candidates_in_image"] = len(candidates)
        queues_by_image.append(selected)
    return [
        image_queue[round_index]
        for round_index in range(
            max((len(items) for items in queues_by_image), default=0)
        )
        for image_queue in queues_by_image
        if round_index < len(image_queue)
    ]


def quality_refill_queue(
    manifest_rows: list[dict[str, str]],
    annotation_dir: Path,
    image_dir: Path,
    output_dir: Path,
    target_complete_per_image: int,
) -> list[dict]:
    """Choose the next-clear candidates for images below a mask target."""
    all_candidates = quality_queue(
        manifest_rows,
        annotation_dir,
        image_dir,
        max_per_image=10**9,
    )
    candidates_by_image: dict[str, list[dict]] = {}
    for item in all_candidates:
        candidates_by_image.setdefault(item["image"], []).append(item)

    completed_by_image: dict[str, int] = {}
    reviewed_ids = set()
    for path in output_dir.glob("*.json"):
        annotation = json.loads(path.read_text())
        object_id = annotation.get("object_id")
        image = annotation.get("image")
        if object_id:
            reviewed_ids.add(object_id)
        if image and annotation.get("status") == "complete":
            completed_by_image[image] = completed_by_image.get(image, 0) + 1

    queues_by_image = []
    for image in sorted(candidates_by_image):
        deficit = max(
            0,
            target_complete_per_image - completed_by_image.get(image, 0),
        )
        available = [
            item for item in candidates_by_image[image]
            if item["object_id"] not in reviewed_ids
        ]
        selected = available[:deficit]
        if len(selected) < deficit:
            raise ValueError(
                f"{image} needs {deficit} additional masks but only "
                f"{len(selected)} unreviewed candidates remain"
            )
        for item in selected:
            item["selection_method"] = "local_image_quality_refill_v1"
            item["refill_target_complete_per_image"] = (
                target_complete_per_image
            )
        if selected:
            queues_by_image.append(selected)
    return [
        image_queue[round_index]
        for round_index in range(
            max((len(items) for items in queues_by_image), default=0)
        )
        for image_queue in queues_by_image
        if round_index < len(image_queue)
    ]


@dataclass
class MaskStore:
    image_dir: Path
    output_dir: Path
    queue: list[dict]
    padding: float = 0.75
    proposal_dir: Path | None = None
    show_proposals: bool = False
    reviewer: str = ""

    def __post_init__(self) -> None:
        self.image_dir = self.image_dir.resolve()
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.proposal_dir is not None:
            self.proposal_dir = self.proposal_dir.resolve()
        self.by_id = {item["object_id"]: item for item in self.queue}
        self._image_cache: dict[str, tuple[bytes, int, int]] = {}

    def image(self, filename: str) -> tuple[bytes, int, int]:
        filename = Path(filename).name
        if filename in self._image_cache:
            return self._image_cache[filename]
        path = self.image_dir / filename
        if not path.is_file():
            raise FileNotFoundError(filename)
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            buffer = io.BytesIO()
            oriented.save(buffer, format="JPEG", quality=96)
            result = (buffer.getvalue(), oriented.width, oriented.height)
        self._image_cache[filename] = result
        return result

    def annotation_path(self, object_id: str) -> Path:
        if object_id not in self.by_id:
            raise ValueError("unknown object_id")
        return self.output_dir / f"{object_id.replace(':', '_')}.json"

    def crop(self, item: dict) -> dict[str, int]:
        _, width, height = self.image(item["image"])
        box = item["box"]
        box_width = box["x2"] - box["x1"]
        box_height = box["y2"] - box["y1"]
        return {
            "x1": max(0, math.floor(box["x1"] - box_width * self.padding)),
            "y1": max(0, math.floor(box["y1"] - box_height * self.padding)),
            "x2": min(width, math.ceil(box["x2"] + box_width * self.padding)),
            "y2": min(height, math.ceil(box["y2"] + box_height * self.padding)),
        }

    def item(self, index: int) -> dict:
        if not (0 <= index < len(self.queue)):
            raise IndexError(index)
        item = dict(self.queue[index])
        item["index"] = index
        item["total"] = len(self.queue)
        item["crop"] = self.crop(item)
        path = self.annotation_path(item["object_id"])
        if path.is_file():
            item["annotation"] = json.loads(path.read_text())
        else:
            proposal_path = (
                self.proposal_dir
                / f"{item['object_id'].replace(':', '_')}.json"
                if self.proposal_dir is not None else None
            )
            proposal = (
                json.loads(proposal_path.read_text())
                if self.show_proposals
                and proposal_path is not None
                and proposal_path.is_file()
                else {}
            )
            item["annotation"] = {
                "status": "not_started",
                "polygon": proposal.get("polygon", []),
                "notes": "",
                "starting_proposal_loaded": bool(proposal.get("polygon")),
            }
            if proposal:
                box = item["box"]
                box_area = (
                    (box["x2"] - box["x1"])
                    * (box["y2"] - box["y1"])
                )
                item["proposal_summary"] = {
                    "model_confidence": proposal.get("model_confidence"),
                    "box_iou_with_expert": proposal.get(
                        "box_iou_with_expert"
                    ),
                    "area_to_box_ratio": (
                        proposal.get("complex_area_px2") / box_area
                        if proposal.get("complex_area_px2") is not None
                        and box_area > 0 else None
                    ),
                }
        return item

    def progress(self) -> dict:
        counts = {
            "complete": 0,
            "needs_review": 0,
            "skipped": 0,
            "not_started": 0,
        }
        for item in self.queue:
            path = self.annotation_path(item["object_id"])
            status = (
                json.loads(path.read_text()).get("status", "not_started")
                if path.is_file() else "not_started"
            )
            counts[status] = counts.get(status, 0) + 1
        return {"total": len(self.queue), **counts}

    def save(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        object_id = str(payload.get("object_id", ""))
        item = self.by_id.get(object_id)
        if item is None:
            raise ValueError("unknown object_id")
        status = str(payload.get("status", "complete"))
        if status not in {"complete", "needs_review", "skipped"}:
            raise ValueError("invalid status")
        _, width, height = self.image(item["image"])
        polygon = []
        if status != "skipped":
            polygon = validate_polygon(payload.get("polygon"), width, height)
        record = {
            "schema_version": 1,
            **item,
            "status": status,
            "polygon": polygon,
            "complex_area_px2": polygon_area(polygon) if polygon else None,
            "notes": str(payload.get("notes", ""))[:1000],
            "annotation_method": "manual_polygon_outer_stomatal_complex",
            "reviewer": self.reviewer,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "starting_proposal_available": (
                self.proposal_dir is not None
                and (
                    self.proposal_dir
                    / f"{object_id.replace(':', '_')}.json"
                ).is_file()
            ),
            "annotation_blinded_to_proposal": not self.show_proposals,
        }
        destination = self.annotation_path(object_id)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            dir=self.output_dir,
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return record


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Stomatal complex masks</title>
<style>
:root{color-scheme:dark;--bg:#0b0d10;--panel:#171a20;--line:#343b47;--mint:#42e3a4;--amber:#ffad33}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:#edf2f8;font:14px system-ui}
body{display:grid;grid-template-rows:62px 1fr 54px}header,footer{display:flex;align-items:center;gap:9px;padding:9px 12px;background:var(--panel)}
header{border-bottom:1px solid var(--line)}footer{border-top:1px solid var(--line);color:#b9c1ce}
button,input{background:#252b35;color:#eef3f9;border:1px solid #414a58;border-radius:6px;padding:8px 11px}
button.primary{background:#176d50;border-color:#269b73}.spacer{flex:1}#stage{position:relative;min-height:0}
canvas{display:block;width:100%;height:100%;cursor:crosshair}#meta{color:#c7cfda}#progress{color:var(--mint);font-weight:700}
kbd{background:#282e37;border:1px solid #434b57;border-bottom-width:2px;border-radius:4px;padding:1px 5px}
</style></head><body>
<header>
<button id="prev">◀</button><button id="next">▶</button>
<span id="meta"></span><span class="spacer"></span><span id="progress"></span>
<input id="notes" placeholder="Optional note"><button id="review">Needs review</button>
<button id="skip">Skip</button><button class="primary" id="save">Save + next</button>
</header>
<main id="stage"><canvas id="canvas"></canvas></main>
<footer>
Click around the <b>outer stomatal complex</b> · double-click or <kbd>Enter</kbd> closes ·
drag a white vertex to correct it · <kbd>A</kbd> accepts · <kbd>V</kbd> flags review ·
<kbd>X</kbd> skips · <kbd>Z</kbd> removes last point · <kbd>R</kbd> resets · wheel zooms
</footer>
<script>
"use strict";
const canvas=document.querySelector("#canvas"),ctx=canvas.getContext("2d"),stage=document.querySelector("#stage");
const $=s=>document.querySelector(s);let index=0,item=null,img=new Image(),points=[],closed=false;
let zoom=1,ox=0,oy=0,space=false,pan=null,dragVertex=-1;
function resize(){let r=stage.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;canvas.style.width=r.width+"px";canvas.style.height=r.height+"px";ctx.setTransform(d,0,0,d,0,0);draw()}
function fit(){if(!item)return;let r=stage.getBoundingClientRect(),c=item.crop,w=c.x2-c.x1,h=c.y2-c.y1;zoom=Math.min(r.width/w,r.height/h)*.92;ox=(r.width-w*zoom)/2-c.x1*zoom;oy=(r.height-h*zoom)/2-c.y1*zoom;draw()}
const screenToImage=(x,y)=>[(x-ox)/zoom,(y-oy)/zoom];const imageToScreen=(x,y)=>[ox+x*zoom,oy+y*zoom];
function draw(){let r=stage.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);if(!item||!img.complete)return;
ctx.drawImage(img,ox,oy,img.naturalWidth*zoom,img.naturalHeight*zoom);
let b=item.box,a=imageToScreen(b.x1,b.y1),z=imageToScreen(b.x2,b.y2);ctx.strokeStyle="#ffad33";ctx.lineWidth=2;ctx.setLineDash([8,5]);ctx.strokeRect(a[0],a[1],z[0]-a[0],z[1]-a[1]);ctx.setLineDash([]);
if(points.length){ctx.beginPath();points.forEach((p,i)=>{let s=imageToScreen(p[0],p[1]);i?ctx.lineTo(...s):ctx.moveTo(...s)});if(closed)ctx.closePath();ctx.fillStyle="rgba(66,227,164,.22)";ctx.fill();ctx.strokeStyle="#42e3a4";ctx.lineWidth=3;ctx.stroke();points.forEach(p=>{let s=imageToScreen(...p);ctx.beginPath();ctx.arc(...s,4,0,Math.PI*2);ctx.fillStyle="#fff";ctx.fill()})}}
async function load(i){try{let response=await fetch("/api/item?index="+i);if(!response.ok)throw new Error(`item request failed (${response.status})`);item=await response.json();index=i;points=item.annotation.polygon||[];closed=points.length>=3;
$("#notes").value=item.annotation.notes||"";let proposal=item.proposal_summary||{},ratio=proposal.area_to_box_ratio==null?"?":proposal.area_to_box_ratio.toFixed(2),qc=proposal.model_confidence==null?"":` · SAM ${proposal.model_confidence.toFixed(2)} · mask/box ${ratio}`;
let quality=item.quality_rank_within_image==null?"":` · quality pick ${item.quality_rank_within_image} · focus ${item.local_focus_score.toFixed(1)}`;
$("#meta").textContent=`${item.image} · ${item.variety} · ${item.water_status} · object ${item.expert_box_index}${quality}${qc}`;
img.onload=()=>fit();img.onerror=()=>{$("#meta").textContent=`Could not load image ${item.image}`};img.src="/api/image?name="+encodeURIComponent(item.image);await progress()}catch(error){console.error(error);$("#meta").textContent=`Load failed: ${error.message}`}}
async function progress(){let response=await fetch("/api/progress");if(!response.ok)throw new Error(`progress request failed (${response.status})`);let p=await response.json();$("#progress").textContent=`${p.complete} complete / ${p.total} · ${p.needs_review} review · ${p.skipped} skipped`}
async function save(status){if(status!=="skipped"&&points.length<3){alert("Trace at least three points first.");return}let body={object_id:item.object_id,status,polygon:points,notes:$("#notes").value};
let r=await fetch("/api/annotation",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});if(!r.ok){alert(await r.text());return}load(Math.min(index+1,item.total-1))}
canvas.addEventListener("mousedown",e=>{if(space||e.button===1){pan={x:e.clientX,y:e.clientY,ox,oy};return}
let hit=points.findIndex(p=>{let s=imageToScreen(...p);return Math.hypot(s[0]-e.offsetX,s[1]-e.offsetY)<=10});
if(hit>=0){dragVertex=hit;return}if(closed){points=[];closed=false}points.push(screenToImage(e.offsetX,e.offsetY));draw()});
canvas.addEventListener("mousemove",e=>{if(dragVertex>=0){points[dragVertex]=screenToImage(e.offsetX,e.offsetY);draw();return}if(pan){ox=pan.ox+e.clientX-pan.x;oy=pan.oy+e.clientY-pan.y;draw()}});
window.addEventListener("mouseup",()=>{pan=null;dragVertex=-1});canvas.addEventListener("dblclick",e=>{e.preventDefault();if(points.length>=3){closed=true;draw()}});
canvas.addEventListener("wheel",e=>{e.preventDefault();let before=screenToImage(e.offsetX,e.offsetY);zoom=Math.max(.05,Math.min(20,zoom*Math.exp(-e.deltaY*.001)));ox=e.offsetX-before[0]*zoom;oy=e.offsetY-before[1]*zoom;draw()},{passive:false});
window.addEventListener("keydown",e=>{if(e.target instanceof HTMLInputElement)return;if(e.code==="Space")space=true;let k=e.key.toLowerCase();
if(k==="z"){points.pop();closed=false;draw()}if(k==="r"){points=[];closed=false;draw()}
if(e.key==="Enter"&&points.length>=3){closed=true;draw()}if(k==="a"||k==="s")save("complete");
if(k==="v")save("needs_review");if(k==="x")save("skipped");if(e.key==="ArrowLeft")load(Math.max(0,index-1));
if(e.key==="ArrowRight")load(Math.min(item.total-1,index+1))});
window.addEventListener("keyup",e=>{if(e.code==="Space")space=false});
$("#prev").onclick=()=>load(Math.max(0,index-1));$("#next").onclick=()=>load(Math.min(item.total-1,index+1));$("#save").onclick=()=>save("complete");$("#review").onclick=()=>save("needs_review");$("#skip").onclick=()=>save("skipped");
new ResizeObserver(resize).observe(stage);load(0);
</script></body></html>"""


def handler_factory(store: MaskStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send_bytes(
                json.dumps(value).encode(),
                "application/json",
                status,
            )

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
                elif parsed.path == "/api/item":
                    index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                    self.send_json(store.item(index))
                elif parsed.path == "/api/progress":
                    self.send_json(store.progress())
                elif parsed.path == "/api/image":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    body, _, _ = store.image(name)
                    self.send_bytes(body, "image/jpeg")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, FileNotFoundError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.send_json(store.save(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("combined_study_manifest.csv"),
    )
    parser.add_argument(
        "--boxes",
        type=Path,
        default=Path("box_annotations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("complex_mask_annotations"),
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=Path(
            "validation_artifacts/complex_mask_proposals_v1/json"
        ),
    )
    parser.add_argument(
        "--show-proposals",
        action="store_true",
        help=(
            "Show SAM contours for assisted correction. Do not use this "
            "mode to create independent validation references."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("random", "quality", "quality-refill"),
        default="random",
        help=(
            "Use random expert boxes for method validation or the clearest "
            "local image regions for a reduced descriptive sample."
        ),
    )
    parser.add_argument("--max-per-image", type=int, default=8)
    parser.add_argument(
        "--target-complete-per-image",
        type=int,
        default=2,
        help=(
            "For quality-refill mode, select enough new candidates to reach "
            "this many completed masks in every image."
        ),
    )
    parser.add_argument(
        "--queue-manifest",
        type=Path,
        default=Path(
            "validation_artifacts/complex_mask_queue_manifest.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reviewer", default=getpass.getuser())
    args = parser.parse_args()
    if args.max_per_image <= 0:
        parser.error("--max-per-image must be positive")
    if args.target_complete_per_image <= 0:
        parser.error("--target-complete-per-image must be positive")
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    if args.selection_mode == "quality-refill":
        queue = quality_refill_queue(
            manifest,
            args.boxes,
            args.images,
            args.output,
            args.target_complete_per_image,
        )
    elif args.selection_mode == "quality":
        queue = quality_queue(
            manifest,
            args.boxes,
            args.images,
            args.max_per_image,
        )
    else:
        queue = balanced_queue(
            manifest,
            args.boxes,
            args.max_per_image,
            args.seed,
        )
    args.queue_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.queue_manifest.write_text(json.dumps({
        "schema_version": 1,
        "selection_mode": args.selection_mode,
        "selection_method": (
            {
                "quality": "local_image_quality_v1",
                "quality-refill": "local_image_quality_refill_v1",
                "random": "deterministic_random_v1",
            }[args.selection_mode]
        ),
        "selection_policy": (
            "Rank within each image using local focus, contrast, clipping, "
            "and edge containment only; do not use object area, mask area, "
            "treatment, variety, or apparent stomatal shape."
        ),
        "max_per_image": args.max_per_image,
        "target_complete_per_image": (
            args.target_complete_per_image
            if args.selection_mode == "quality-refill" else None
        ),
        "images": len({item["image"] for item in queue}),
        "objects": len(queue),
        "queue": queue,
    }, indent=2) + "\n")
    store = MaskStore(
        args.images,
        args.output,
        queue,
        proposal_dir=args.proposals,
        show_proposals=args.show_proposals,
        reviewer=args.reviewer,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        handler_factory(store),
    )
    print(
        f"Serving {len(queue)} {args.selection_mode}-selected objects at "
        f"http://127.0.0.1:{args.port}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
