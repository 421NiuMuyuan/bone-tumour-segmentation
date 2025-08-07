# convert_labels.py

import os, json
from PIL import Image, ImageDraw

# 自动定位本脚本所在目录
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, "bones-annotated")
OUT_SINGLE  = os.path.join(DATA_ROOT, "masks_single")
OUT_MULTI   = os.path.join(DATA_ROOT, "masks_multi")

# Label → 类别编号映射
LABEL_MAP = {
    "Apophysis":       1,
    "Epiphysis":       2,
    "Metaphysis":      3,
    "Diaphysis":       4,
    "Surface Tumour":  5,
    "In-Bone Tumour":  6,
    "Joint":           7,
}

# 确保输出目录存在
os.makedirs(OUT_SINGLE, exist_ok=True)
for cls_name in LABEL_MAP:
    os.makedirs(os.path.join(OUT_MULTI, cls_name), exist_ok=True)

def find_image(name):
    for sub in ("2","3"):
        p = os.path.join(DATA_ROOT, sub, name)
        if os.path.isfile(p):
            return p
    return None

def main():
    jsons = [f for f in os.listdir(DATA_ROOT) if f.endswith(".json")]
    print(f"Found {len(jsons)} JSON files:", jsons)
    for jf in jsons:
        data = json.load(open(os.path.join(DATA_ROOT, jf), encoding="utf-8"))
        print(f"-- Processing {jf}, {len(data)} items")
        for item in data:
            ref = item["data"].get("image") or item["data"].get("image_url")
            name = os.path.basename(ref)
            img_path = find_image(name)
            if not img_path:
                print("   ⚠ skip", name)
                continue

            img = Image.open(img_path)
            W,H = img.size

            # 单通道 mask
            m_single = Image.new("L", (W,H), 0)
            draw_s = ImageDraw.Draw(m_single)
            # 多通道：为每个 LABEL_MAP 键创建二值画笔
            m_multi = {lbl: Image.new("1", (W,H), 0) for lbl in LABEL_MAP}
            draw_m = {lbl: ImageDraw.Draw(m_multi[lbl]) for lbl in LABEL_MAP}

            count = 0
            for ann in item.get("annotations", []):
                for r in ann.get("result", []):
                    if r.get("type")!="polygonlabels":
                        continue
                    lbl_list = r["value"].get("polygonlabels", [])
                    if not lbl_list:
                        continue
                    lbl = lbl_list[0]
                    if lbl not in LABEL_MAP:
                        continue
                    cls = LABEL_MAP[lbl]
                    pts = r["value"]["points"]
                    poly = [(x*W/100.0, y*H/100.0) for x,y in pts]

                    # 单通道填值
                    draw_s.polygon(poly, fill=cls)
                    # 多通道二值填充
                    draw_m[lbl].polygon(poly, fill=1)
                    count += 1

            base = os.path.splitext(name)[0]
            # 保存 single
            m_single.save(os.path.join(OUT_SINGLE, f"{base}.png"))
            # 保存 multi
            for lbl in LABEL_MAP:
                m_multi[lbl].save(
                    os.path.join(OUT_MULTI, lbl, f"{base}.png")
                )

            print(f"   ✔ {name}: {count} polys → single + multi")

    print("Done. Single masks in", OUT_SINGLE)
    print("Multi masks in", OUT_MULTI)

if __name__=="__main__":
    main()
