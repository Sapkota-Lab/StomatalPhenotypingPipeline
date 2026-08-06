#!/usr/bin/env python3
"""Local, zoomable bounding-box annotator for stomata.

Run:
    .venv/bin/python box_annotator.py

Then open http://127.0.0.1:8765. Annotations are stored as one JSON file per
image under ``box_annotations/``. Images are normalized for EXIF orientation
before display, so saved coordinates match the counting engines.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def safe_image_name(value: str) -> str:
    """Return a plain image filename or raise for traversal/unsupported input."""
    name = Path(value).name
    if not name or name != value or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Invalid image filename")
    return name


def validate_boxes(boxes: object, width: int, height: int) -> list[dict[str, float]]:
    """Validate and normalize browser-supplied boxes."""
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    clean: list[dict[str, float]] = []
    for raw in boxes:
        if not isinstance(raw, dict):
            raise ValueError("each box must be an object")
        try:
            x1, y1, x2, y2 = (float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("box coordinates must be numeric") from exc
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("box is outside the oriented image or has zero area")
        clean.append(
            {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
    return clean


@dataclass
class AnnotationStore:
    image_dir: Path
    output_dir: Path
    priority_csv: Path | None = None

    def __post_init__(self) -> None:
        self.image_dir = self.image_dir.resolve()
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._image_cache: dict[str, tuple[bytes, int, int]] = {}

    def image_names(self) -> list[str]:
        available = {
            path.name
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if self.priority_csv and self.priority_csv.is_file():
            with self.priority_csv.open(newline="") as handle:
                priority = [
                    row["filename"]
                    for row in csv.DictReader(handle)
                    if row.get("filename") in available
                ]
            remainder = sorted(available.difference(priority))
            return priority + remainder
        return sorted(available)

    def oriented_image(self, name: str) -> tuple[bytes, int, int]:
        name = safe_image_name(name)
        if name in self._image_cache:
            return self._image_cache[name]
        path = self.image_dir / name
        if not path.is_file():
            raise FileNotFoundError(name)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=95, optimize=True)
        result = (buffer.getvalue(), width, height)
        self._image_cache[name] = result
        return result

    def annotation_path(self, name: str) -> Path:
        name = safe_image_name(name)
        return self.output_dir / f"{Path(name).stem}.json"

    def load(self, name: str) -> dict:
        name = safe_image_name(name)
        _, width, height = self.oriented_image(name)
        path = self.annotation_path(name)
        if not path.is_file():
            return {
                "image": name,
                "width": width,
                "height": height,
                "boxes": [],
                "status": "not_started",
                "notes": "",
            }
        with path.open() as handle:
            payload = json.load(handle)
        payload["boxes"] = validate_boxes(payload.get("boxes", []), width, height)
        payload["image"] = name
        payload["width"] = width
        payload["height"] = height
        payload.setdefault("status", "in_progress")
        payload.setdefault("notes", "")
        return payload

    def save(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        name = safe_image_name(str(payload.get("image", "")))
        _, width, height = self.oriented_image(name)
        boxes = validate_boxes(payload.get("boxes"), width, height)
        status = str(payload.get("status", "in_progress"))
        if status not in {
            "not_started",
            "in_progress",
            "provisional",
            "complete",
            "excluded",
        }:
            raise ValueError("invalid annotation status")
        notes = str(payload.get("notes", ""))[:2000]
        record = {
            "schema_version": 1,
            "image": name,
            "width": width,
            "height": height,
            "status": status,
            "count": len(boxes),
            "boxes": boxes,
            "notes": notes,
        }
        destination = self.annotation_path(name)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".tmp", dir=self.output_dir
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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stomata box annotator</title>
<style>
:root { color-scheme: dark; --panel:#171a20; --line:#303641; --accent:#42e3a4; }
* { box-sizing:border-box; }
html,body { margin:0; width:100%; height:100%; overflow:hidden; background:#090b0f; color:#e8edf5; font:14px/1.35 system-ui,sans-serif; }
body { display:grid; grid-template-rows:56px 1fr 46px; }
header,footer { display:flex; align-items:center; gap:10px; padding:8px 12px; background:var(--panel); border-color:var(--line); }
header { border-bottom:1px solid var(--line); }
footer { border-top:1px solid var(--line); color:#aeb8c7; }
button,select,input { background:#242a34; color:#eef3fa; border:1px solid #3b4452; border-radius:6px; padding:7px 10px; }
button:hover { border-color:#6c7a8f; }
button.primary { background:#176d50; border-color:#269b73; }
#stage { position:relative; min-height:0; }
canvas { display:block; width:100%; height:100%; cursor:crosshair; }
.spacer { flex:1; }
#count { color:var(--accent); font-weight:750; font-size:17px; min-width:84px; }
#saveState { min-width:68px; color:#9eabbc; }
#notes { width:230px; }
kbd { background:#282e37; border:1px solid #434b57; border-bottom-width:2px; border-radius:4px; padding:1px 5px; color:#dbe3ee; }
</style>
</head>
<body>
<header>
  <button id="prev">◀ Prev</button>
  <select id="imageSelect"></select>
  <button id="next">Next ▶</button>
  <button id="zoomOut">−</button>
  <button id="fit">Fit</button>
  <button id="zoomIn">+</button>
  <span id="zoomLabel">100%</span>
  <span class="spacer"></span>
  <span id="count">0 stomata</span>
  <select id="status">
    <option value="not_started">Not started</option>
    <option value="in_progress">In progress</option>
    <option value="provisional">Codex provisional</option>
    <option value="complete">Complete</option>
    <option value="excluded">Exclude image</option>
  </select>
  <input id="notes" placeholder="Image-quality notes">
  <button class="primary" id="save">Save</button>
  <span id="saveState">Saved</span>
</header>
<main id="stage"><canvas id="canvas"></canvas></main>
<footer>
  Drag to box a stoma · click a box to select · <kbd>Delete</kbd> removes ·
  <kbd>Space</kbd>+drag or middle-drag pans · wheel zooms image ·
  <kbd>S</kbd> saves · <kbd>F</kbd> fits · <kbd>Z</kbd> undoes · <kbd>N/P</kbd> navigates
</footer>
<script>
"use strict";
const canvas=document.querySelector("#canvas"), ctx=canvas.getContext("2d");
const stage=document.querySelector("#stage"), select=document.querySelector("#imageSelect");
let names=[], index=0, image=new Image(), boxes=[], history=[], selected=-1;
let zoom=1, offsetX=0, offsetY=0, drag=null, space=false, dirty=false, saveTimer=null;
const $=s=>document.querySelector(s);

function resize(){ const r=stage.getBoundingClientRect(), d=devicePixelRatio||1;
  canvas.width=Math.round(r.width*d); canvas.height=Math.round(r.height*d);
  canvas.style.width=r.width+"px"; canvas.style.height=r.height+"px";
  ctx.setTransform(d,0,0,d,0,0); draw();
}
function viewSize(){ const r=stage.getBoundingClientRect(); return [r.width,r.height]; }
function fit(){ if(!image.naturalWidth)return; const [w,h]=viewSize();
  zoom=Math.min(w/image.naturalWidth,h/image.naturalHeight);
  offsetX=(w-image.naturalWidth*zoom)/2; offsetY=(h-image.naturalHeight*zoom)/2; updateZoom(); draw();
}
function screenToImage(x,y){ return [(x-offsetX)/zoom,(y-offsetY)/zoom]; }
function imageToScreen(x,y){ return [offsetX+x*zoom,offsetY+y*zoom]; }
function clampBox(b){ return {x1:Math.max(0,Math.min(b.x1,b.x2)),y1:Math.max(0,Math.min(b.y1,b.y2)),
  x2:Math.min(image.naturalWidth,Math.max(b.x1,b.x2)),y2:Math.min(image.naturalHeight,Math.max(b.y1,b.y2))}; }
function updateZoom(){ $("#zoomLabel").textContent=Math.round(zoom*100)+"%"; }
function setDirty(v=true){
  dirty=v; $("#saveState").textContent=v?"Unsaved":"Saved";
  if(saveTimer)clearTimeout(saveTimer);
  saveTimer=v?setTimeout(()=>save().catch(()=>{}),900):null;
}
function snapshot(){ history.push(JSON.stringify(boxes)); if(history.length>100)history.shift(); }
function undo(){ if(!history.length)return; boxes=JSON.parse(history.pop()); selected=-1; setDirty(); updateCount(); draw(); }
function updateCount(){ $("#count").textContent=boxes.length+" "+(boxes.length===1?"stoma":"stomata"); }

function draw(){
  const [w,h]=viewSize(); ctx.clearRect(0,0,w,h); ctx.fillStyle="#090b0f";ctx.fillRect(0,0,w,h);
  if(!image.naturalWidth)return;
  ctx.imageSmoothingEnabled=zoom<1;ctx.drawImage(image,offsetX,offsetY,image.naturalWidth*zoom,image.naturalHeight*zoom);
  ctx.font="600 13px system-ui";ctx.textBaseline="bottom";
  boxes.forEach((b,i)=>{ const [x1,y1]=imageToScreen(b.x1,b.y1),[x2,y2]=imageToScreen(b.x2,b.y2);
    ctx.lineWidth=i===selected?3:2;ctx.strokeStyle=i===selected?"#ffe167":"#42e3a4";ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    const label=String(i+1);ctx.fillStyle=i===selected?"#ffe167":"#42e3a4";ctx.fillRect(x1,y1-20,Math.max(24,8+ctx.measureText(label).width),20);
    ctx.fillStyle="#07100c";ctx.fillText(label,x1+5,y1-3);
  });
  if(drag?.mode==="box"){ const [x1,y1]=imageToScreen(drag.x1,drag.y1),[x2,y2]=imageToScreen(drag.x2,drag.y2);
    ctx.setLineDash([7,5]);ctx.lineWidth=2;ctx.strokeStyle="#ffe167";ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
  }
}
function canvasPoint(e){ const r=canvas.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top]; }
function hitBox(ix,iy){ for(let i=boxes.length-1;i>=0;i--){const b=boxes[i];if(ix>=b.x1&&ix<=b.x2&&iy>=b.y1&&iy<=b.y2)return i;}return -1; }
canvas.addEventListener("pointerdown",e=>{ canvas.setPointerCapture(e.pointerId); const [sx,sy]=canvasPoint(e);
  if(e.button===1||space){drag={mode:"pan",sx,sy,ox:offsetX,oy:offsetY};canvas.style.cursor="grabbing";return;}
  if(e.button!==0)return; const [ix,iy]=screenToImage(sx,sy), hit=hitBox(ix,iy);
  if(hit>=0){selected=hit;draw();return;} selected=-1;drag={mode:"box",x1:ix,y1:iy,x2:ix,y2:iy};draw();
});
canvas.addEventListener("pointermove",e=>{ if(!drag)return;const [sx,sy]=canvasPoint(e);
  if(drag.mode==="pan"){offsetX=drag.ox+sx-drag.sx;offsetY=drag.oy+sy-drag.sy;}
  else {[drag.x2,drag.y2]=screenToImage(sx,sy);} draw();
});
canvas.addEventListener("pointerup",e=>{ if(!drag)return;
  if(drag.mode==="box"){const b=clampBox(drag);if((b.x2-b.x1)>=8/zoom&&(b.y2-b.y1)>=8/zoom){snapshot();boxes.push(b);selected=boxes.length-1;if($("#status").value==="not_started")$("#status").value="in_progress";setDirty();updateCount();}}
  drag=null;canvas.style.cursor="crosshair";draw();
});
canvas.addEventListener("wheel",e=>{e.preventDefault();const [sx,sy]=canvasPoint(e),[ix,iy]=screenToImage(sx,sy);
  zoom=Math.max(.05,Math.min(16,zoom*(e.deltaY<0?1.18:1/1.18)));offsetX=sx-ix*zoom;offsetY=sy-iy*zoom;updateZoom();draw();
},{passive:false});

async function loadCurrent(){
  if(dirty)await save();
  const name=names[index];select.value=name;$("#saveState").textContent="Loading…";
  const data=await fetch("/api/annotations?name="+encodeURIComponent(name)).then(r=>r.json());
  boxes=data.boxes||[];history=[];selected=-1;$("#status").value=data.status||"not_started";$("#notes").value=data.notes||"";
  image=new Image();image.onload=()=>{fit();updateCount();setDirty(false);};image.src="/api/image?name="+encodeURIComponent(name);
}
async function save(){
  if(!names.length)return;if(saveTimer)clearTimeout(saveTimer);saveTimer=null;$("#saveState").textContent="Saving…";
  const body={image:names[index],boxes,status:$("#status").value,notes:$("#notes").value};
  const r=await fetch("/api/annotations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(!r.ok){$("#saveState").textContent="Save failed";throw new Error(await r.text());}setDirty(false);
}
async function navigate(delta){index=(index+delta+names.length)%names.length;await loadCurrent();}
function removeSelected(){if(selected<0)return;snapshot();boxes.splice(selected,1);selected=-1;setDirty();updateCount();draw();}
function zoomCenter(factor){const [w,h]=viewSize(),[ix,iy]=screenToImage(w/2,h/2);zoom=Math.max(.05,Math.min(16,zoom*factor));offsetX=w/2-ix*zoom;offsetY=h/2-iy*zoom;updateZoom();draw();}

$("#prev").onclick=()=>navigate(-1);$("#next").onclick=()=>navigate(1);$("#save").onclick=save;$("#fit").onclick=fit;
$("#zoomIn").onclick=()=>zoomCenter(1.25);$("#zoomOut").onclick=()=>zoomCenter(.8);
select.onchange=()=>{index=names.indexOf(select.value);loadCurrent();};
$("#status").onchange=()=>setDirty();$("#notes").oninput=()=>setDirty();
addEventListener("keydown",e=>{if(e.code==="Space"){space=true;e.preventDefault();}
  if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName))return;
  if(e.key==="Delete"||e.key==="Backspace"){removeSelected();e.preventDefault();}
  else if(e.key.toLowerCase()==="s"){save();e.preventDefault();}
  else if(e.key.toLowerCase()==="f"){fit();}
  else if(e.key.toLowerCase()==="z"){undo();}
  else if(e.key.toLowerCase()==="n"){navigate(1);}
  else if(e.key.toLowerCase()==="p"){navigate(-1);}
  else if(e.key==="+"||e.key==="="){zoomCenter(1.25);}
  else if(e.key==="-"){zoomCenter(.8);}
});
addEventListener("keyup",e=>{if(e.code==="Space")space=false;});
addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
new ResizeObserver(resize).observe(stage);
(async()=>{names=await fetch("/api/images").then(r=>r.json());for(const n of names){const o=document.createElement("option");o.value=o.textContent=n;select.append(o);}await loadCurrent();})();
</script>
</body>
</html>"""


def make_handler(store: AnnotationStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def send_json(self, payload: object, status: int = 200) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    encoded = HTML.encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                elif parsed.path == "/api/images":
                    self.send_json(store.image_names())
                elif parsed.path == "/api/image":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    encoded, width, height = store.oriented_image(name)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("X-Image-Width", str(width))
                    self.send_header("X-Image-Height", str(height))
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                elif parsed.path == "/api/annotations":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    self.send_json(store.load(name))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/annotations":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2_000_000:
                    raise ValueError("request body is too large")
                payload = json.loads(self.rfile.read(length))
                self.send_json(store.save(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("Stomata_Images"))
    parser.add_argument("--output", type=Path, default=Path("box_annotations"))
    parser.add_argument(
        "--priority",
        type=Path,
        default=Path("annotation_priority_target_v1.csv"),
        help="Optional CSV whose filename column controls initial image order",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    store = AnnotationStore(args.images, args.output, args.priority)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    url = f"http://{args.host}:{args.port}"
    print(f"Stomata box annotator: {url}")
    print("Press Ctrl-C to stop. Annotations save under", store.output_dir)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
