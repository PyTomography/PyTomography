"""Build docs/source/_static/memory/gpus.json, the GPU list of the "Memory and speed" page's calculator, from a
hand-checked table, then check that every renderer string finds its own entry.

No network access: the table below is the data. To update it, edit the rows of G (each row cites the page its
memory size was read from), run `python docs/tools/build_gpus.py`, and check that it reports 0 problems (exit code 0).
No downloadable GPU database is used: none gives memory sizes with sources and the renderer strings browsers report.

Every number below was read from the page given as its source (WebFetch, 11 Oct 2026).
Entries are ordered so that a page scanning top to bottom can take the FIRST match: an entry matches when its
`match`, or one of its `aliases` (other forms of the same renderer string, e.g. Linux's Mesa form), is a substring of
the lower-cased WebGL renderer string. Firefox's stand-ins (strings ending ", or similar") come first: Firefox reports
one of a few bucket models instead of the real GPU, so they say which vendor it is, not which GPU.
"""
import json
import os
import sys

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "_static", "memory")

# ---- sources -------------------------------------------------------------
NV_COMPARE = "https://www.nvidia.com/en-us/geforce/graphics-cards/compare/"
NV_LAPTOPS = "https://www.nvidia.com/en-us/geforce/laptops/compare/"
NV_30_LAPTOPS = "https://www.nvidia.com/en-us/geforce/laptops/30-series/"
NV_5090DV2 = "https://www.nvidia.cn/geforce/graphics-cards/50-series/rtx-5090-d-v2/"
WIKI_20 = "https://en.wikipedia.org/wiki/GeForce_RTX_20_series"
WIKI_30 = "https://en.wikipedia.org/wiki/GeForce_RTX_30_series"
NV_PRO_DESK = "https://www.nvidia.com/en-us/design-visualization/desktop-graphics/"
NV_PRO_LAPTOP = "https://www.nvidia.com/en-gb/products/workstations/professional-laptops/compare/"
WIKI_QUADRO = "https://en.wikipedia.org/wiki/Quadro"
HOTHW_A_LAPTOP = "https://hothardware.com/news/nvidia-rtx-a5500-workstation-gpu-pro-workloads"
NV_H100 = "https://www.nvidia.com/en-us/data-center/h100/"
NV_H200 = "https://www.nvidia.com/en-us/data-center/h200/"
NV_A100 = "https://www.nvidia.com/en-us/data-center/a100/"
NV_L4 = "https://www.nvidia.com/en-us/data-center/l4/"
NV_L40S = "https://www.nvidia.com/en-us/data-center/l40s/"
NV_L40 = "https://www.nvidia.com/en-us/data-center/l40/"
NV_T4 = "https://www.nvidia.com/en-us/data-center/tesla-t4/"
NV_A10 = "https://www.nvidia.com/en-us/data-center/products/a10-gpu/"
NV_A40 = "https://www.nvidia.com/en-us/data-center/a40/"
NV_DGX_B200 = "https://www.nvidia.com/en-us/data-center/dgx-b200/"
NV_DGX_B300 = "https://www.nvidia.com/en-us/data-center/dgx-b300/"
NV_GB200 = "https://www.nvidia.com/en-us/data-center/gb200-nvl72/"
NV_DGX_SPARK = "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
AWS_G5 = "https://aws.amazon.com/ec2/instance-types/g5/"
GCP_GPUS = "https://docs.cloud.google.com/compute/docs/gpus"
WIKI_TESLA = "https://en.wikipedia.org/wiki/Nvidia_Tesla"
GETDEPLOYING = "https://getdeploying.com/dataset/gpu-models/gpu-models.csv"
ROCM = "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html"
AMD_MI325X = "https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi325x.html"
AMD_MI300X = "https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi300x.html"
AMD_MI355X = "https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html"
WIKI_RX9000 = "https://en.wikipedia.org/wiki/Radeon_RX_9000_series"
WIKI_RX7000 = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_RX_7000"
WIKI_RX7000M = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_RX_7000M"
WIKI_RX6000 = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_RX_6000"
WIKI_RX6000M = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_RX_6000M"
WIKI_W7000 = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_Pro_W7000"
WIKI_W6000 = "https://en.wikipedia.org/wiki/Template:AMD_Radeon_Pro_W6000"
AP_M1 = "https://support.apple.com/en-us/111883"
AP_M1PRO = "https://support.apple.com/en-us/111902"
AP_M1ULTRA = "https://support.apple.com/en-us/111900"
AP_M2 = "https://support.apple.com/en-us/111867"
AP_M2PRO = "https://support.apple.com/en-us/111340"
AP_M2ULTRA = "https://support.apple.com/en-us/111835"
AP_M3 = "https://support.apple.com/en-us/118551"
AP_M3PRO = "https://support.apple.com/en-us/117736"
AP_M3ULTRA = "https://support.apple.com/en-us/122211"
AP_M4 = "https://support.apple.com/en-us/121555"
AP_M4MAX = "https://support.apple.com/en-us/121553"
AP_MBP = "https://www.apple.com/macbook-pro/specs/"
AP_STUDIO = "https://www.apple.com/mac-studio/specs/"
AP_M6 = "https://support.apple.com/en-us/128108"
AP_NEO = "https://support.apple.com/en-us/126322"
WIKI_ARC_A = "https://en.wikipedia.org/wiki/Template:Intel_Arc_Alchemist"
WIKI_ARC_AM = "https://en.wikipedia.org/wiki/Template:Intel_Arc_Alchemist_Mobile"
WIKI_ARC_AW = "https://en.wikipedia.org/wiki/Template:Intel_Arc_Alchemist_Workstation"
WIKI_ARC_B = "https://en.wikipedia.org/wiki/Template:Intel_Arc_Battlemage"
WIKI_ARC_BW = "https://en.wikipedia.org/wiki/Template:Intel_Arc_Battlemage_Workstation"
INTEL_A770 = "https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html"
INTEL_B580 = "https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html"
WIKI_PTL = "https://en.wikipedia.org/wiki/Panther_Lake_(microprocessor)"
W3DS_INTEL = "https://web3dsurvey.com/webgl/vendors/Google%20Inc.%20(Intel)"
W3DS_AMD = "https://web3dsurvey.com/webgl/vendors/Google%20Inc.%20(AMD)"
W3DS_APPLE = "https://web3dsurvey.com/webgl/vendors/Google%20Inc.%20(Apple)"
FX_SANITIZE = "https://searchfox.org/mozilla-central/source/dom/canvas/SanitizeRenderer.cpp"
FX_BUG_1916271 = "https://bugzilla.mozilla.org/show_bug.cgi?id=1916271"

NV, AMD, AP, IN = "NVIDIA", "AMD", "Apple", "Intel"
D, DC, U = "discrete", "datacenter", "unified"
SHARED = "integrated; shares system RAM (no dedicated memory)"
FX = ("Firefox reports one of a few stand-in models (here '{}, or similar') instead of the real GPU, so the model and "
      "its memory are unknown: pick your GPU or its memory")

# (match, name, vendor, memory_gb, kind, notes, source)
G = [
    # ---------------- Firefox stand-ins (keep first): SanitizeRenderer.cpp reports a bucket model + ", or similar" ----------------
    ("geforce gtx 980, or similar", "NVIDIA GPU (Firefox shows a stand-in)", NV, None, D, FX.format("GeForce GTX 980"), FX_SANITIZE),
    ("radeon r9 200 series, or similar", "AMD GPU (Firefox shows a stand-in)", AMD, None, D, FX.format("Radeon R9 200 Series"), FX_SANITIZE),
    ("intel(r) hd graphics, or similar", "Intel GPU (Firefox shows a stand-in)", IN, None, U, FX.format("Intel(R) HD Graphics"), FX_SANITIZE),
    ("apple m1, or similar", "Apple GPU (Firefox shows a stand-in)", AP, None, U,
     "Firefox reports every Apple chip as 'Apple M1, or similar': pick your chip or its memory", FX_SANITIZE),
    (", or similar", "GPU (Firefox shows a stand-in)", "unknown", None, D, FX.format("..."), FX_SANITIZE),
    # ---------------- NVIDIA GeForce RTX 50 ----------------
    ("rtx 5090 laptop", "GeForce RTX 5090 Laptop GPU", NV, 24, D, "laptop", NV_LAPTOPS),
    ("rtx 5080 laptop", "GeForce RTX 5080 Laptop GPU", NV, 16, D, "laptop", NV_LAPTOPS),
    ("rtx 5070 ti laptop", "GeForce RTX 5070 Ti Laptop GPU", NV, 12, D, "laptop", NV_LAPTOPS),
    ("rtx 5070 laptop", "GeForce RTX 5070 Laptop GPU", NV, [12, 8], D, "laptop; NVIDIA lists 12 GB and 8 GB versions", NV_LAPTOPS),
    ("rtx 5060 laptop", "GeForce RTX 5060 Laptop GPU", NV, 8, D, "laptop", NV_LAPTOPS),
    ("rtx 5050 laptop", "GeForce RTX 5050 Laptop GPU", NV, 8, D, "laptop", NV_LAPTOPS),
    ("rtx 5090 d v2", "GeForce RTX 5090 D v2", NV, 24, D, "China-market card", NV_5090DV2),
    ("rtx 5090", "GeForce RTX 5090", NV, 32, D, "the China-market RTX 5090 D also matches here", NV_COMPARE),
    ("rtx 5080", "GeForce RTX 5080", NV, 16, D, None, NV_COMPARE),
    ("rtx 5070 ti", "GeForce RTX 5070 Ti", NV, 16, D, None, NV_COMPARE),
    ("rtx 5070", "GeForce RTX 5070", NV, 12, D, None, NV_COMPARE),
    ("rtx 5060 ti", "GeForce RTX 5060 Ti", NV, [16, 8], D, "16 GB and 8 GB versions; the renderer string does not say which", NV_COMPARE),
    ("rtx 5060", "GeForce RTX 5060", NV, 8, D, None, NV_COMPARE),
    ("rtx 5050", "GeForce RTX 5050", NV, 8, D, None, NV_COMPARE),
    # ---------------- NVIDIA GeForce RTX 40 ----------------
    ("rtx 4090 laptop", "GeForce RTX 4090 Laptop GPU", NV, 16, D, "laptop", NV_LAPTOPS),
    ("rtx 4080 laptop", "GeForce RTX 4080 Laptop GPU", NV, 12, D, "laptop", NV_LAPTOPS),
    ("rtx 4070 laptop", "GeForce RTX 4070 Laptop GPU", NV, 8, D, "laptop", NV_LAPTOPS),
    ("rtx 4060 laptop", "GeForce RTX 4060 Laptop GPU", NV, 8, D, "laptop", NV_LAPTOPS),
    ("rtx 4050 laptop", "GeForce RTX 4050 Laptop GPU", NV, 6, D, "laptop", NV_LAPTOPS),
    ("rtx 4090", "GeForce RTX 4090", NV, 24, D, "the China-market RTX 4090 D (also 24 GB) matches here too", NV_COMPARE),
    ("rtx 4080 super", "GeForce RTX 4080 SUPER", NV, 16, D, None, NV_COMPARE),
    ("rtx 4080", "GeForce RTX 4080", NV, 16, D, None, NV_COMPARE),
    ("rtx 4070 ti super", "GeForce RTX 4070 Ti SUPER", NV, 16, D, None, NV_COMPARE),
    ("rtx 4070 ti", "GeForce RTX 4070 Ti", NV, 12, D, None, NV_COMPARE),
    ("rtx 4070 super", "GeForce RTX 4070 SUPER", NV, 12, D, None, NV_COMPARE),
    ("rtx 4070", "GeForce RTX 4070", NV, 12, D, None, NV_COMPARE),
    ("rtx 4060 ti", "GeForce RTX 4060 Ti", NV, [16, 8], D, "16 GB and 8 GB versions; the renderer string does not say which", NV_COMPARE),
    ("rtx 4060", "GeForce RTX 4060", NV, 8, D, None, NV_COMPARE),
    # ---------------- NVIDIA GeForce RTX 30 ----------------
    ("rtx 3080 ti laptop", "GeForce RTX 3080 Ti Laptop GPU", NV, 16, D, "laptop", NV_30_LAPTOPS),
    ("rtx 3080 laptop", "GeForce RTX 3080 Laptop GPU", NV, [16, 8], D, "laptop; 16 GB and 8 GB versions", NV_30_LAPTOPS),
    ("rtx 3070 ti laptop", "GeForce RTX 3070 Ti Laptop GPU", NV, 8, D, "laptop", NV_30_LAPTOPS),
    ("rtx 3070 laptop", "GeForce RTX 3070 Laptop GPU", NV, 8, D, "laptop", NV_30_LAPTOPS),
    ("rtx 3060 laptop", "GeForce RTX 3060 Laptop GPU", NV, 6, D, "laptop", NV_30_LAPTOPS),
    ("rtx 3050 ti laptop", "GeForce RTX 3050 Ti Laptop GPU", NV, 4, D, "laptop", NV_30_LAPTOPS),
    ("rtx 3050 6gb laptop", "GeForce RTX 3050 6GB Laptop GPU", NV, 6, D, "laptop; renderer name assumed from NVIDIA's '6GB' product naming, not seen in a renderer list", WIKI_30),
    ("rtx 3050 laptop", "GeForce RTX 3050 Laptop GPU", NV, [4, 6], D, "laptop; 4 GB and 6 GB versions (RTX 3050 A Laptop GPU, 4 GB, is not covered)", NV_30_LAPTOPS),
    ("rtx 3090 ti", "GeForce RTX 3090 Ti", NV, 24, D, None, NV_COMPARE),
    ("rtx 3090", "GeForce RTX 3090", NV, 24, D, None, NV_COMPARE),
    ("rtx 3080 ti", "GeForce RTX 3080 Ti", NV, 12, D, None, NV_COMPARE),
    ("rtx 3080", "GeForce RTX 3080", NV, [12, 10], D, "12 GB and 10 GB versions", NV_COMPARE),
    ("rtx 3070 ti", "GeForce RTX 3070 Ti", NV, 8, D, None, NV_COMPARE),
    ("rtx 3070", "GeForce RTX 3070", NV, 8, D, None, NV_COMPARE),
    ("rtx 3060 ti", "GeForce RTX 3060 Ti", NV, 8, D, None, NV_COMPARE),
    ("rtx 3060", "GeForce RTX 3060", NV, [12, 8], D, "12 GB and 8 GB versions", NV_COMPARE),
    ("rtx 3050", "GeForce RTX 3050", NV, [8, 6], D, "8 GB and 6 GB versions", NV_COMPARE),
    # ---------------- NVIDIA GeForce RTX 20 / TITAN / GTX 16 ----------------
    ("rtx 2080 ti", "GeForce RTX 2080 Ti", NV, 11, D, None, NV_COMPARE),
    ("rtx 2080 super", "GeForce RTX 2080 SUPER", NV, 8, D, "the laptop part has the same name and 8 GB", NV_COMPARE),
    ("rtx 2080", "GeForce RTX 2080", NV, 8, D, "the laptop part (incl. Max-Q) has the same name and 8 GB", NV_COMPARE),
    ("rtx 2070 super", "GeForce RTX 2070 SUPER", NV, 8, D, "the laptop part has the same name and 8 GB", NV_COMPARE),
    ("rtx 2070", "GeForce RTX 2070", NV, 8, D, "the laptop part (incl. Max-Q) has the same name and 8 GB", NV_COMPARE),
    ("rtx 2060 super", "GeForce RTX 2060 SUPER", NV, 8, D, None, NV_COMPARE),
    ("rtx 2060 12gb", "GeForce RTX 2060 12GB", NV, 12, D, "renderer name assumed ('12GB' suffix), not seen in a renderer list", WIKI_20),
    ("rtx 2060", "GeForce RTX 2060", NV, [6, 12], D, "6 GB and 12 GB versions; the laptop RTX 2060 (same name) has 6 GB", NV_COMPARE),
    ("rtx 2050", "GeForce RTX 2050", NV, 4, D, "laptop-only part (name has no 'Laptop')", WIKI_20),
    ("titan rtx", "NVIDIA TITAN RTX", NV, 24, D, None, WIKI_20),
    ("gtx 1660 ti", "GeForce GTX 1660 Ti", NV, 6, D, None, NV_COMPARE),
    ("gtx 1660 super", "GeForce GTX 1660 SUPER", NV, 6, D, None, NV_COMPARE),
    ("gtx 1660", "GeForce GTX 1660", NV, 6, D, None, NV_COMPARE),
    ("gtx 1650 super", "GeForce GTX 1650 SUPER", NV, 4, D, None, NV_COMPARE),
    ("gtx 1650", "GeForce GTX 1650", NV, 4, D, None, NV_COMPARE),
    ("gtx 1630", "GeForce GTX 1630", NV, 4, D, None, NV_COMPARE),
    # ---------------- NVIDIA RTX PRO Blackwell ----------------
    ("rtx pro 6000 blackwell", "NVIDIA RTX PRO 6000 Blackwell (Workstation, Max-Q, Server Edition)", NV, 96, D, "workstation; Server Edition also 96 GB", NV_PRO_DESK),
    ("rtx pro 5500 blackwell", "NVIDIA RTX PRO 5500 Blackwell", NV, 84, D, "workstation", NV_PRO_DESK),
    ("rtx pro 5000 blackwell generation laptop", "NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU", NV, 24, D, "workstation laptop; renderer name assumed by analogy with 'RTX 500 Ada Generation Laptop GPU'", NV_PRO_LAPTOP),
    ("rtx pro 5000 blackwell", "NVIDIA RTX PRO 5000 Blackwell", NV, [48, 72], D, "workstation; 48 GB and 72 GB versions", NV_PRO_DESK),
    ("rtx pro 4500 blackwell", "NVIDIA RTX PRO 4500 Blackwell", NV, 32, D, "workstation", NV_PRO_DESK),
    ("rtx pro 4000 blackwell generation laptop", "NVIDIA RTX PRO 4000 Blackwell Generation Laptop GPU", NV, 16, D, "workstation laptop; renderer name assumed", NV_PRO_LAPTOP),
    ("rtx pro 4000 blackwell", "NVIDIA RTX PRO 4000 Blackwell (incl. SFF Edition)", NV, 24, D, "workstation", NV_PRO_DESK),
    ("rtx pro 3000 blackwell", "NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU", NV, 12, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    ("rtx pro 2000 blackwell generation laptop", "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU", NV, 8, D, "workstation laptop; renderer name assumed", NV_PRO_LAPTOP),
    ("rtx pro 2000 blackwell", "NVIDIA RTX PRO 2000 Blackwell", NV, 16, D, "workstation", NV_PRO_DESK),
    ("rtx pro 1000 blackwell", "NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU", NV, 8, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    ("rtx pro 500 blackwell", "NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU", NV, 6, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    # ---------------- NVIDIA RTX Ada Generation ----------------
    ("rtx 6000 ada", "NVIDIA RTX 6000 Ada Generation", NV, 48, D, "workstation", NV_PRO_DESK),
    ("rtx 5880 ada", "NVIDIA RTX 5880 Ada Generation", NV, 48, D, "workstation", WIKI_QUADRO),
    ("rtx 5000 ada generation laptop", "NVIDIA RTX 5000 Ada Generation Laptop GPU", NV, 16, D, "workstation laptop", NV_PRO_LAPTOP),
    ("rtx 5000 ada", "NVIDIA RTX 5000 Ada Generation", NV, 32, D, "workstation", NV_PRO_DESK),
    ("rtx 4500 ada", "NVIDIA RTX 4500 Ada Generation", NV, 24, D, "workstation", NV_PRO_DESK),
    ("rtx 4000 ada generation laptop", "NVIDIA RTX 4000 Ada Generation Laptop GPU", NV, 12, D, "workstation laptop", NV_PRO_LAPTOP),
    ("rtx 4000 sff ada", "NVIDIA RTX 4000 SFF Ada Generation", NV, 20, D, "workstation", NV_PRO_DESK),
    ("rtx 4000 ada", "NVIDIA RTX 4000 Ada Generation", NV, 20, D, "workstation", NV_PRO_DESK),
    ("rtx 3500 ada", "NVIDIA RTX 3500 Ada Generation Laptop GPU", NV, 12, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    ("rtx 3000 ada", "NVIDIA RTX 3000 Ada Generation Laptop GPU", NV, 8, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    ("rtx 2000 ada generation laptop", "NVIDIA RTX 2000 Ada Generation Laptop GPU", NV, 8, D, "workstation laptop", NV_PRO_LAPTOP),
    ("rtx 2000 ada", "NVIDIA RTX 2000 Ada Generation", NV, 16, D, "workstation", NV_PRO_DESK),
    ("rtx 1000 ada", "NVIDIA RTX 1000 Ada Generation Laptop GPU", NV, 6, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    ("rtx 500 ada", "NVIDIA RTX 500 Ada Generation Laptop GPU", NV, 4, D, "workstation laptop (laptop-only part)", NV_PRO_LAPTOP),
    # ---------------- NVIDIA RTX A-series (Ampere) ----------------
    ("rtx a6000", "NVIDIA RTX A6000", NV, 48, D, "workstation", NV_PRO_DESK),
    ("rtx a5500 laptop", "NVIDIA RTX A5500 Laptop GPU", NV, 16, D, "workstation laptop", HOTHW_A_LAPTOP),
    ("rtx a5500", "NVIDIA RTX A5500", NV, 24, D, "workstation", NV_PRO_DESK),
    ("rtx a5000 laptop", "NVIDIA RTX A5000 Laptop GPU", NV, 16, D, "workstation laptop", WIKI_QUADRO),
    ("rtx a5000", "NVIDIA RTX A5000", NV, 24, D, "workstation", NV_PRO_DESK),
    ("rtx a4500 laptop", "NVIDIA RTX A4500 Laptop GPU", NV, 16, D, "workstation laptop", HOTHW_A_LAPTOP),
    ("rtx a4500", "NVIDIA RTX A4500", NV, 20, D, "workstation", NV_PRO_DESK),
    ("rtx a4000 laptop", "NVIDIA RTX A4000 Laptop GPU", NV, 8, D, "workstation laptop", WIKI_QUADRO),
    ("rtx a4000", "NVIDIA RTX A4000", NV, 16, D, "workstation", NV_PRO_DESK),
    ("rtx a3000 12gb laptop", "NVIDIA RTX A3000 12GB Laptop GPU", NV, 12, D, "workstation laptop; renderer name assumed", HOTHW_A_LAPTOP),
    ("rtx a3000 laptop", "NVIDIA RTX A3000 Laptop GPU", NV, [6, 12], D, "workstation laptop; original 6 GB, later 12 GB version", WIKI_QUADRO),
    ("rtx a2000 8gb laptop", "NVIDIA RTX A2000 8GB Laptop GPU", NV, 8, D, "workstation laptop; renderer name assumed", HOTHW_A_LAPTOP),
    ("rtx a2000 laptop", "NVIDIA RTX A2000 Laptop GPU", NV, [4, 8], D, "workstation laptop; original 4 GB, later 8 GB version", WIKI_QUADRO),
    ("rtx a2000 12gb", "NVIDIA RTX A2000 12GB", NV, 12, D, "workstation", NV_PRO_DESK),
    ("rtx a2000", "NVIDIA RTX A2000", NV, [6, 12], D, "workstation; 6 GB and 12 GB versions", NV_PRO_DESK),
    ("rtx a1000 laptop", "NVIDIA RTX A1000 Laptop GPU", NV, 4, D, "workstation laptop", HOTHW_A_LAPTOP),
    ("rtx a1000", "NVIDIA RTX A1000", NV, 8, D, "workstation", NV_PRO_DESK),
    ("rtx a500 laptop", "NVIDIA RTX A500 Laptop GPU", NV, 4, D, "workstation laptop", HOTHW_A_LAPTOP),
    ("rtx a400", "NVIDIA RTX A400", NV, 4, D, "workstation", NV_PRO_DESK),
    # ---------------- NVIDIA Quadro RTX / T-series (Turing) ----------------
    ("quadro rtx 8000", "Quadro RTX 8000", NV, 48, D, "workstation", WIKI_QUADRO),
    ("quadro rtx 6000", "Quadro RTX 6000", NV, 24, D, "workstation; the mobile part is also 24 GB", WIKI_QUADRO),
    ("quadro rtx 5000", "Quadro RTX 5000", NV, 16, D, "workstation; the mobile part (incl. Max-Q) is also 16 GB", WIKI_QUADRO),
    ("quadro rtx 4000", "Quadro RTX 4000", NV, 8, D, "workstation; the mobile part is also 8 GB", WIKI_QUADRO),
    ("quadro rtx 3000", "Quadro RTX 3000", NV, 6, D, "workstation laptop (mobile-only part)", WIKI_QUADRO),
    ("quadro t2000", "Quadro T2000", NV, 4, D, "workstation laptop", WIKI_QUADRO),
    ("quadro t1000", "Quadro T1000", NV, 4, D, "workstation laptop", WIKI_QUADRO),
    ("nvidia t1000 8gb", "NVIDIA T1000 8GB", NV, 8, D, "workstation", NV_PRO_DESK),
    ("nvidia t1000", "NVIDIA T1000", NV, [4, 8], D, "workstation; 4 GB and 8 GB versions", NV_PRO_DESK),
    ("nvidia t600", "NVIDIA T600", NV, 4, D, "workstation", NV_PRO_DESK),
    ("nvidia t400 4gb", "NVIDIA T400 4GB", NV, 4, D, "workstation", NV_PRO_DESK),
    ("nvidia t400", "NVIDIA T400", NV, [2, 4], D, "workstation; 2 GB and 4 GB versions", NV_PRO_DESK),
    # ---------------- NVIDIA data center / cloud ----------------
    ("gb300", "NVIDIA GB300 (Grace Blackwell Ultra)", NV, None, DC, "not confirmed: Google Cloud a4x-maxgpu-4g = 1,116 GB for 4 GPUs (279 GB each); getdeploying lists 288 GB", GCP_GPUS),
    ("b300", "NVIDIA B300 (Blackwell Ultra)", NV, None, DC, "not confirmed: DGX B300 = 2.1 TB for 8 GPUs; getdeploying lists 288 GB", NV_DGX_B300),
    ("gb200", "NVIDIA GB200 (Grace Blackwell)", NV, 186, DC, "per GPU: GB200 superchip = 372 GB for 2 GPUs; NVL72 = 13.4 TB for 72 GPUs", NV_GB200),
    ("b200", "NVIDIA B200", NV, 180, DC, "per GPU: DGX B200 = 1,440 GB for 8 GPUs; 192 GB was quoted at the 2024 announcement", NV_DGX_B200),
    ("b100", "NVIDIA B100", NV, 192, DC, None, WIKI_TESLA),
    ("gh200", "NVIDIA GH200 (Grace Hopper)", NV, 96, DC, "96 GB HBM3 GPU memory per getdeploying; a later version has more (not confirmed here)", GETDEPLOYING),
    ("h200", "NVIDIA H200 (SXM and NVL)", NV, 141, DC, None, NV_H200),
    ("h100 nvl", "NVIDIA H100 NVL", NV, 94, DC, "94 GB usable; 188 GB for an NVLink pair", NV_H100),
    ("h100", "NVIDIA H100 (SXM, PCIe)", NV, 80, DC, "SXM and PCIe are 80 GB; H100 NVL is 94 GB; Wikipedia also lists 64 GB and 96 GB SXM variants", NV_H100),
    ("nvidia a100", "NVIDIA A100", NV, [80, 40], DC, "80 GB and 40 GB versions (nvidia-smi names include 40GB/80GB)", GCP_GPUS),
    ("a10g", "NVIDIA A10G (AWS G5)", NV, 24, DC, "cloud (AWS)", AWS_G5),
    ("nvidia a10", "NVIDIA A10", NV, 24, DC, None, NV_A10),
    ("nvidia a30", "NVIDIA A30", NV, 24, DC, None, WIKI_TESLA),
    ("nvidia a40", "NVIDIA A40", NV, 48, DC, None, NV_A40),
    ("nvidia l40s", "NVIDIA L40S", NV, 48, DC, None, NV_L40S),
    ("nvidia l40", "NVIDIA L40", NV, 48, DC, None, NV_L40),
    ("nvidia l4", "NVIDIA L4", NV, 24, DC, "common Colab GPU", NV_L4),
    ("tesla t4", "NVIDIA T4", NV, 16, DC, "common Colab GPU (free tier)", NV_T4),
    ("nvidia t4", "NVIDIA T4", NV, 16, DC, "common Colab GPU (free tier)", NV_T4),
    ("v100", "NVIDIA V100", NV, [32, 16], DC, "32 GB and 16 GB versions; Google Cloud's V100 is 16 GB", WIKI_TESLA),
    ("p100", "NVIDIA P100", NV, [16, 12], DC, "16 GB and 12 GB (PCIe) versions", WIKI_TESLA),
    ("nvidia gb10", "NVIDIA GB10 (DGX Spark and OEM versions)", NV, [128, 64], U, "unified LPDDR5X shared by CPU and GPU; 64 GB only from OEM partners", NV_DGX_SPARK),
    # ---------------- AMD Instinct ----------------
    ("mi355x", "AMD Instinct MI355X", AMD, 288, DC, None, AMD_MI355X),
    ("mi350x", "AMD Instinct MI350X", AMD, 288, DC, None, ROCM),
    ("mi325x", "AMD Instinct MI325X", AMD, 256, DC, "first announced with 288 GB, shipped with 256 GB", AMD_MI325X),
    ("mi300x", "AMD Instinct MI300X", AMD, 192, DC, None, AMD_MI300X),
    ("mi300a", "AMD Instinct MI300A", AMD, 128, DC, "APU: the 128 GB HBM3 is shared by CPU and GPU", ROCM),
    ("mi250x", "AMD Instinct MI250X", AMD, 128, DC, "two GCDs; each appears as a separate 64 GB device", ROCM),
    ("mi250", "AMD Instinct MI250", AMD, 128, DC, "two GCDs; each appears as a separate 64 GB device", ROCM),
    ("mi210", "AMD Instinct MI210", AMD, 64, DC, None, ROCM),
    ("mi100", "AMD Instinct MI100", AMD, 32, DC, None, ROCM),
    # ---------------- AMD Radeon AI PRO / PRO W ----------------
    ("ai pro r9700", "AMD Radeon AI PRO R9700 (incl. R9700S)", AMD, 32, D, "workstation", ROCM),
    ("ai pro r9600d", "AMD Radeon AI PRO R9600D", AMD, 32, D, "workstation", ROCM),
    ("radeon pro w7900", "AMD Radeon PRO W7900 (incl. Dual Slot)", AMD, 48, D, "workstation", ROCM),
    ("radeon pro w7800 48gb", "AMD Radeon PRO W7800 48GB", AMD, 48, D, "workstation", ROCM),
    ("radeon pro w7800", "AMD Radeon PRO W7800", AMD, 32, D, "workstation", ROCM),
    ("radeon pro w7700", "AMD Radeon PRO W7700", AMD, 16, D, "workstation", ROCM),
    ("radeon pro w7600", "AMD Radeon PRO W7600", AMD, 8, D, "workstation", WIKI_W7000),
    ("radeon pro w7500", "AMD Radeon PRO W7500", AMD, 8, D, "workstation", WIKI_W7000),
    ("radeon pro w6800", "AMD Radeon PRO W6800", AMD, 32, D, "workstation", WIKI_W6000),
    ("radeon pro w6600", "AMD Radeon PRO W6600", AMD, 8, D, "workstation", WIKI_W6000),
    ("radeon pro w6400", "AMD Radeon PRO W6400", AMD, 4, D, "workstation", WIKI_W6000),
    ("radeon pro w6300", "AMD Radeon PRO W6300", AMD, 2, D, "workstation", WIKI_W6000),
    # ---------------- AMD Radeon RX 9000 ----------------
    ("rx 9070 xt", "AMD Radeon RX 9070 XT", AMD, 16, D, None, ROCM),
    ("rx 9070 gre", "AMD Radeon RX 9070 GRE", AMD, 12, D, "ROCm's table says 16 GB; Wikipedia and launch coverage say 12 GB (192-bit)", WIKI_RX9000),
    ("rx 9070", "AMD Radeon RX 9070", AMD, 16, D, None, ROCM),
    ("rx 9060 xt", "AMD Radeon RX 9060 XT", AMD, [16, 8], D, "16 GB and 8 GB versions", WIKI_RX9000),
    ("rx 9060", "AMD Radeon RX 9060", AMD, 8, D, None, ROCM),
    ("rx 9050", "AMD Radeon RX 9050", AMD, [8, 4], D, "8 GB and 4 GB versions", ROCM),
    # ---------------- AMD Radeon RX 7000 ----------------
    ("rx 7900 xtx", "AMD Radeon RX 7900 XTX", AMD, 24, D, None, ROCM),
    ("rx 7900 xt", "AMD Radeon RX 7900 XT", AMD, 20, D, None, ROCM),
    ("rx 7900 gre", "AMD Radeon RX 7900 GRE", AMD, 16, D, None, ROCM),
    ("rx 7900m", "AMD Radeon RX 7900M", AMD, 16, D, "laptop", WIKI_RX7000M),
    ("rx 7800 xt", "AMD Radeon RX 7800 XT", AMD, 16, D, None, ROCM),
    ("rx 7800m", "AMD Radeon RX 7800M", AMD, 12, D, "laptop", WIKI_RX7000M),
    ("rx 7700 xt", "AMD Radeon RX 7700 XT", AMD, 12, D, None, ROCM),
    ("rx 7700s", "AMD Radeon RX 7700S", AMD, 8, D, "laptop", WIKI_RX7000M),
    ("rx 7700", "AMD Radeon RX 7700", AMD, 16, D, None, ROCM),
    ("rx 7650 gre", "AMD Radeon RX 7650 GRE", AMD, 8, D, None, WIKI_RX7000),
    ("rx 7600m xt", "AMD Radeon RX 7600M XT", AMD, 8, D, "laptop", WIKI_RX7000M),
    ("rx 7600m", "AMD Radeon RX 7600M", AMD, 8, D, "laptop", WIKI_RX7000M),
    ("rx 7600s", "AMD Radeon RX 7600S", AMD, 8, D, "laptop", WIKI_RX7000M),
    ("rx 7600 xt", "AMD Radeon RX 7600 XT", AMD, 16, D, None, WIKI_RX7000),
    ("rx 7600", "AMD Radeon RX 7600", AMD, 8, D, None, ROCM),
    ("rx 7400", "AMD Radeon RX 7400", AMD, 8, D, None, WIKI_RX7000),
    # ---------------- AMD Radeon RX 6000 ----------------
    ("rx 6950 xt", "AMD Radeon RX 6950 XT", AMD, 16, D, None, WIKI_RX6000),
    ("rx 6900 xt", "AMD Radeon RX 6900 XT", AMD, 16, D, None, WIKI_RX6000),
    ("rx 6850m xt", "AMD Radeon RX 6850M XT", AMD, 12, D, "laptop", WIKI_RX6000M),
    ("rx 6800 xt", "AMD Radeon RX 6800 XT", AMD, 16, D, None, WIKI_RX6000),
    ("rx 6800m", "AMD Radeon RX 6800M", AMD, 12, D, "laptop", WIKI_RX6000M),
    ("rx 6800s", "AMD Radeon RX 6800S", AMD, 8, D, "laptop", WIKI_RX6000M),
    ("rx 6800", "AMD Radeon RX 6800", AMD, 16, D, None, WIKI_RX6000),
    ("rx 6750 gre 10gb", "AMD Radeon RX 6750 GRE 10GB", AMD, 10, D, "renderer name assumed", WIKI_RX6000),
    ("rx 6750 gre", "AMD Radeon RX 6750 GRE", AMD, [12, 10], D, "12 GB and 10 GB versions", WIKI_RX6000),
    ("rx 6750 xt", "AMD Radeon RX 6750 XT", AMD, 12, D, None, WIKI_RX6000),
    ("rx 6700 xt", "AMD Radeon RX 6700 XT", AMD, 12, D, None, WIKI_RX6000),
    ("rx 6700m", "AMD Radeon RX 6700M", AMD, 10, D, "laptop", WIKI_RX6000M),
    ("rx 6700s", "AMD Radeon RX 6700S", AMD, 8, D, "laptop", WIKI_RX6000M),
    ("rx 6700", "AMD Radeon RX 6700", AMD, 10, D, None, WIKI_RX6000),
    ("rx 6650m xt", "AMD Radeon RX 6650M XT", AMD, 8, D, "laptop", WIKI_RX6000M),
    ("rx 6650m", "AMD Radeon RX 6650M", AMD, 8, D, "laptop", WIKI_RX6000M),
    ("rx 6650 xt", "AMD Radeon RX 6650 XT", AMD, 8, D, None, WIKI_RX6000),
    ("rx 6600 xt", "AMD Radeon RX 6600 XT", AMD, 8, D, None, WIKI_RX6000),
    ("rx 6600m", "AMD Radeon RX 6600M", AMD, 8, D, "laptop", WIKI_RX6000M),
    ("rx 6600s", "AMD Radeon RX 6600S", AMD, 4, D, "laptop; Wikipedia table rowspan looked inconsistent here", WIKI_RX6000M),
    ("rx 6600", "AMD Radeon RX 6600", AMD, 8, D, None, WIKI_RX6000),
    ("rx 6550m", "AMD Radeon RX 6550M", AMD, 4, D, "laptop", WIKI_RX6000M),
    ("rx 6550s", "AMD Radeon RX 6550S", AMD, 4, D, "laptop", WIKI_RX6000M),
    ("rx 6500m", "AMD Radeon RX 6500M", AMD, 4, D, "laptop", WIKI_RX6000M),
    ("rx 6500 xt", "AMD Radeon RX 6500 XT", AMD, [4, 8], D, "4 GB and 8 GB versions", WIKI_RX6000),
    ("rx 6450m", "AMD Radeon RX 6450M", AMD, 4, D, "laptop", WIKI_RX6000M),
    ("rx 6400", "AMD Radeon RX 6400", AMD, 4, D, None, WIKI_RX6000),
    ("rx 6300m", "AMD Radeon RX 6300M", AMD, 2, D, "laptop", WIKI_RX6000M),
    ("rx 6300", "AMD Radeon RX 6300", AMD, 2, D, None, WIKI_RX6000),
    # ---------------- AMD integrated (unified) ----------------
    ("8060s graphics", "AMD Radeon 8060S (Ryzen AI Max, Strix Halo)", AMD, None, U, SHARED, W3DS_AMD),
    ("8050s graphics", "AMD Radeon 8050S (Ryzen AI Max, Strix Halo)", AMD, None, U, SHARED, W3DS_AMD),
    ("890m graphics", "AMD Radeon 890M", AMD, None, U, SHARED, W3DS_AMD),
    ("880m graphics", "AMD Radeon 880M", AMD, None, U, SHARED, W3DS_AMD),
    ("780m graphics", "AMD Radeon 780M", AMD, None, U, SHARED, W3DS_AMD),
    ("760m graphics", "AMD Radeon 760M", AMD, None, U, SHARED, W3DS_AMD),
    # ---------------- Apple silicon (unified) ----------------
    ("apple m6", "Apple M6", AP, [16, 24, 32], U, "unified memory (Mac mini 2026)", AP_M6),
    ("apple m5 ultra", "Apple M5 Ultra", AP, [96, 256, 512], U, "unified memory (Mac Studio)", AP_STUDIO),
    ("apple m5 max", "Apple M5 Max", AP, [36, 48, 64, 128], U, "unified memory", AP_MBP),
    ("apple m5 pro", "Apple M5 Pro", AP, [24, 48, 64], U, "unified memory", AP_MBP),
    ("apple m5", "Apple M5", AP, [16, 24, 32], U, "unified memory (Mac); iPad Pro M5 is 12 or 16 GB per Wikipedia, but Safari on iPad reports only 'Apple GPU'", AP_MBP),
    ("apple m4 max", "Apple M4 Max", AP, [36, 48, 64, 128], U, "unified memory", AP_M4MAX),
    ("apple m4 pro", "Apple M4 Pro", AP, [24, 48, 64], U, "unified memory", AP_M4),
    ("apple m4", "Apple M4", AP, [16, 24, 32], U, "unified memory (Mac); iPad Pro M4 is 8 or 16 GB per Wikipedia", AP_M4),
    ("apple m3 ultra", "Apple M3 Ultra", AP, [96, 256, 512], U, "unified memory; Apple's Mac Studio (2025) page now lists 96/256 GB; 512 GB was offered at launch (Wikipedia)", AP_M3ULTRA),
    ("apple m3 max", "Apple M3 Max", AP, [36, 48, 64, 96, 128], U, "unified memory", AP_M3PRO),
    ("apple m3 pro", "Apple M3 Pro", AP, [18, 36], U, "unified memory", AP_M3PRO),
    ("apple m3", "Apple M3", AP, [8, 16, 24], U, "unified memory", AP_M3),
    ("apple m2 ultra", "Apple M2 Ultra", AP, [64, 128, 192], U, "unified memory", AP_M2ULTRA),
    ("apple m2 max", "Apple M2 Max", AP, [32, 64, 96], U, "unified memory", AP_M2PRO),
    ("apple m2 pro", "Apple M2 Pro", AP, [16, 32], U, "unified memory", AP_M2PRO),
    ("apple m2", "Apple M2", AP, [8, 16, 24], U, "unified memory", AP_M2),
    ("apple m1 ultra", "Apple M1 Ultra", AP, [64, 128], U, "unified memory", AP_M1ULTRA),
    ("apple m1 max", "Apple M1 Max", AP, [32, 64], U, "unified memory", AP_M1PRO),
    ("apple m1 pro", "Apple M1 Pro", AP, [16, 32], U, "unified memory", AP_M1PRO),
    ("apple m1", "Apple M1", AP, [8, 16], U, "unified memory", AP_M1),
    ("apple a18 pro", "Apple A18 Pro (MacBook Neo)", AP, 8, U, "unified memory (MacBook Neo)", AP_NEO),
    # ---------------- Intel Arc discrete ----------------
    ("arc(tm) pro b70", "Intel Arc Pro B70", IN, 32, D, "workstation", WIKI_ARC_BW),
    ("arc(tm) pro b60", "Intel Arc Pro B60", IN, 24, D, "workstation", WIKI_ARC_BW),
    ("arc(tm) pro b50", "Intel Arc Pro B50", IN, 16, D, "workstation", WIKI_ARC_BW),
    ("arc(tm) pro a60m", "Intel Arc Pro A60M", IN, 8, D, "workstation laptop", WIKI_ARC_AW),
    ("arc(tm) pro a60", "Intel Arc Pro A60", IN, 12, D, "workstation", WIKI_ARC_AW),
    ("arc(tm) pro a50", "Intel Arc Pro A50", IN, 6, D, "workstation", WIKI_ARC_AW),
    ("arc(tm) pro a40", "Intel Arc Pro A40", IN, 6, D, "workstation", WIKI_ARC_AW),
    ("arc(tm) pro a30m", "Intel Arc Pro A30M", IN, 4, D, "workstation laptop", WIKI_ARC_AW),
    ("arc(tm) b580", "Intel Arc B580", IN, 12, D, None, INTEL_B580),
    ("arc(tm) b570", "Intel Arc B570", IN, 10, D, None, WIKI_ARC_B),
    ("arc(tm) a770m", "Intel Arc A770M", IN, 16, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a730m", "Intel Arc A730M", IN, 12, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a570m", "Intel Arc A570M", IN, 8, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a550m", "Intel Arc A550M", IN, 8, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a530m", "Intel Arc A530M", IN, [4, 8], D, "laptop; 4 GB and 8 GB versions", WIKI_ARC_AM),
    ("arc(tm) a370m", "Intel Arc A370M", IN, 4, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a350m", "Intel Arc A350M", IN, 4, D, "laptop", WIKI_ARC_AM),
    ("arc(tm) a770", "Intel Arc A770", IN, [16, 8], D, "16 GB and 8 GB versions; the renderer string does not say which", INTEL_A770),
    ("arc(tm) a750", "Intel Arc A750", IN, 8, D, None, WIKI_ARC_A),
    ("arc(tm) a580", "Intel Arc A580", IN, 8, D, None, WIKI_ARC_A),
    ("arc(tm) a380", "Intel Arc A380", IN, 6, D, None, WIKI_ARC_A),
    ("arc(tm) a310", "Intel Arc A310", IN, 4, D, None, WIKI_ARC_A),
    # ---------------- Intel integrated (unified) ----------------
    ("arc(tm) b390", "Intel Arc B390 (Panther Lake integrated)", IN, None, U, SHARED, WIKI_PTL),
    ("arc(tm) b370", "Intel Arc B370 (Panther Lake integrated)", IN, None, U, SHARED, WIKI_PTL),
    ("arc(tm) 140v", "Intel Arc 140V (Lunar Lake integrated)", IN, None, U, SHARED + "; the renderer string ends in the system RAM, e.g. '(16GB)'", W3DS_INTEL),
    ("arc(tm) 130v", "Intel Arc 130V (Lunar Lake integrated)", IN, None, U, SHARED + "; the renderer string ends in the system RAM, e.g. '(8GB)'", W3DS_INTEL),
    ("arc(tm) 140t", "Intel Arc 140T (Arrow Lake integrated)", IN, None, U, SHARED + "; the renderer string ends in the system RAM, e.g. '(16GB)'", W3DS_INTEL),
    ("arc(tm) 130t", "Intel Arc 130T (Arrow Lake integrated)", IN, None, U, SHARED, W3DS_INTEL),
    ("arc(tm) pro graphics", "Intel Arc Pro Graphics (Core Ultra integrated)", IN, None, U, SHARED, W3DS_INTEL),
    ("intel(r) arc(tm) graphics", "Intel Arc Graphics (Core Ultra integrated)", IN, None, U, SHARED, W3DS_INTEL),
    ("iris(r) xe", "Intel Iris Xe Graphics", IN, None, U, SHARED, W3DS_INTEL),
    ("iris(r) plus", "Intel Iris Plus Graphics", IN, None, U, SHARED, W3DS_INTEL),
    ("iris(tm) plus", "Intel Iris Plus Graphics", IN, None, U, SHARED, W3DS_INTEL),
    ("uhd graphics", "Intel UHD Graphics", IN, None, U, SHARED, W3DS_INTEL),
    ("intel(r) hd graphics", "Intel HD Graphics", IN, None, U, SHARED, W3DS_INTEL),
    ("intel(r) graphics", "Intel Graphics (integrated)", IN, None, U, SHARED, W3DS_INTEL),
    # ---------------- generic fallbacks (keep last) ----------------
    ("radeon(tm)", "AMD Radeon integrated graphics", AMD, None, U, SHARED + "; generic fallback for 'AMD Radeon(TM) ... Graphics' APUs", W3DS_AMD),
    ("radeon graphics", "AMD Radeon integrated graphics", AMD, None, U, SHARED + "; generic fallback (Linux/Mesa form)", W3DS_AMD),
    ("apple gpu", "Apple GPU (Safari hides the chip)", AP, None, U, "Safari/WebKit reports only 'Apple GPU'; ask the user which chip and memory", W3DS_APPLE),
]

KEYS = ("match", "name", "vendor", "memory_gb", "kind", "notes", "source")

# Other forms of an entry's renderer string, by the entry's match. Firefox's buckets (SanitizeRenderer.cpp) in their
# plain form and their Windows (ANGLE Direct3D) form; Linux's Mesa form of Intel Xe graphics.
D3D_FX = " direct3d11 vs_5_0 ps_5_0), or similar"
ALIASES = {
    "geforce gtx 980, or similar": ["geforce gtx 980" + D3D_FX, "geforce gtx 480, or similar", "geforce gtx 480" + D3D_FX,
                                    "geforce 8800 gtx, or similar", "geforce 8800 gtx" + D3D_FX, "nvidia tegra, or similar"],
    "radeon r9 200 series, or similar": ["radeon r9 200 series" + D3D_FX, "radeon hd 5850, or similar", "radeon hd 5850" + D3D_FX,
                                         "radeon hd 3200 graphics, or similar", "radeon hd 3200 graphics" + D3D_FX],
    "intel(r) hd graphics, or similar": ["intel(r) hd graphics" + D3D_FX, "intel(r) hd graphics 400, or similar",
                                         "intel(r) hd graphics 400" + D3D_FX, "intel(r) arc(tm) a750 graphics, or similar",
                                         "intel(r) arc(tm) a750 graphics" + D3D_FX, "intel 945gm, or similar", "intel 945gm" + D3D_FX],
    "apple m1, or similar": ["apple m1, unspecified version), or similar"],
    ", or similar": ["generic renderer"],
    "iris(r) xe": ["intel(r) xe graphics"],
}


def templates(m):
    """Plausible renderer strings containing match m, in several browser forms."""
    return [
        f"angle (x, x {m} (0x00001234) direct3d11 vs_5_0 ps_5_0, d3d11)",
        f"angle (x, x {m} direct3d11 vs_5_0 ps_5_0, d3d11)",
        f"angle (x, x {m} gpu (0x00001234) direct3d11 vs_5_0 ps_5_0, d3d11)",
        f"angle (x, angle metal renderer: {m}, unspecified version)",
        f"angle (x, vulkan 1.4.341 (nvidia {m} (0x00001234)), nvidia)",
        f"angle (x, mesa {m} (radeonsi navi31 aco), opengl es 3.2)",
    ]


# Real strings seen on web3dsurvey.com (lower-cased) -> expected match
REAL = {
    "angle (nvidia, nvidia geforce rtx 4090 (0x00002684) direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 4090",
    "angle (nvidia, nvidia geforce rtx 4060 laptop gpu (0x000028e0) direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 4060 laptop",
    "angle (nvidia, nvidia geforce rtx 3060 laptop direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 3060 laptop",
    "angle (nvidia, nvidia geforce rtx 4060 direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 4060",
    "angle (nvidia, nvidia geforce rtx 5060 ti (0x00002d04) direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 5060 ti",
    "angle (nvidia, nvidia geforce rtx 4070 super (0x00002783) direct3d11 vs_5_0 ps_5_0, d3d11)": "rtx 4070 super",
    "angle (nvidia, vulkan 1.4.341 (nvidia nvidia rtx 500 ada generation laptop gpu (0x000028ba)), nvidia)": "rtx 500 ada",
    "angle (nvidia, vulkan 1.4.351 (nvidia nvidia geforce rtx 5060 (0x00002d05)), nvidia)": "rtx 5060",
    "angle (nvidia, vulkan 1.4.312 (nvidia nvidia gb10 (0x00002e12)), nvidia)": "nvidia gb10",
    "angle (nvidia, nvidia quadro t2000 (0x00001fb8) direct3d11 vs_5_0 ps_5_0, d3d11)": "quadro t2000",
    "angle (amd, amd radeon rx 7900 xt (radeonsi navi31 aco), opengl es 3.2)": "rx 7900 xt",
    "angle (amd, amd radeon rx 7900 xtx (radeonsi navi31 aco), opengl es 3.2)": "rx 7900 xtx",
    "angle (amd, amd radeon rx 9070 xt (0x00007550) direct3d11 vs_5_0 ps_5_0, d3d11)": "rx 9070 xt",
    "angle (amd, amd radeon rx 6700 xt (0x000073df) direct3d11 vs_5_0 ps_5_0, d3d11)": "rx 6700 xt",
    "angle (amd, amd radeon rx 6600 (0x000073ff) direct3d11 vs_5_0 ps_5_0, d3d11)": "rx 6600",
    "angle (amd, amd radeon pro w6400 (radeonsi navi24 aco), opengl es 3.2)": "radeon pro w6400",
    "angle (amd, amd radeon(tm) graphics (0x00001638) direct3d11 vs_5_0 ps_5_0, d3d11)": "radeon(tm)",
    "angle (amd, amd radeon 780m graphics (0x00001900) direct3d11 vs_5_0 ps_5_0, d3d11)": "780m graphics",
    "angle (amd, amd radeon(tm) 890m graphics (0x0000150e) direct3d11 vs_5_0 ps_5_0, d3d11)": "890m graphics",
    "angle (amd, radeon 8060s graphics (radeonsi gfx1151 llvm 20.1.2), opengl es 3.2)": "8060s graphics",
    "angle (amd, amd radeon graphics (radeonsi renoir aco), opengl es 3.2)": "radeon graphics",
    "angle (apple, angle metal renderer: apple m2 pro, unspecified version)": "apple m2 pro",
    "angle (apple, angle metal renderer: apple m1, unspecified version)": "apple m1",
    "angle (apple, angle metal renderer: apple m5 max, unspecified version)": "apple m5 max",
    "angle (apple, angle metal renderer: apple a18 pro, unspecified version)": "apple a18 pro",
    "angle (intel, intel(r) arc(tm) a750 graphics direct3d11 vs_5_0 ps_5_0), or similar": "intel(r) hd graphics, or similar",
    "angle (intel, intel(r) arc(tm) graphics (0x00007d55) direct3d11 vs_5_0 ps_5_0, d3d11)": "intel(r) arc(tm) graphics",
    "angle (intel, intel(r) arc(tm) pro graphics (0x00007d55) direct3d11 vs_5_0 ps_5_0, d3d11)": "arc(tm) pro graphics",
    "angle (intel, mesa intel(r) arc(tm) graphics (mtl), opengl es 3.2)": "intel(r) arc(tm) graphics",
    "angle (intel, intel(r) arc(tm) 140v gpu (16gb) (0x000064a0) direct3d11 vs_5_0 ps_5_0, d3d11)": "arc(tm) 140v",
    "angle (intel, intel(r) iris(r) xe graphics (0x00009a49) direct3d11 vs_5_0 ps_5_0, d3d11)": "iris(r) xe",
    "angle (intel, intel(r) uhd graphics 770 (0x00004680) direct3d11 vs_5_0 ps_5_0, d3d11)": "uhd graphics",
    "angle (intel, mesa intel(r) graphics (arl), opengl es 3.2)": "intel(r) graphics",
    "angle (intel, angle metal renderer: intel(r) iris(tm) plus graphics, unspecified version)": "iris(tm) plus",
}


# Other strings (lower-cased) -> expected match: Firefox's output quoted in bug 1916271 and its bucket forms, and the
# Linux (Mesa) form of Intel Xe graphics that Chrome reports.
OTHER = {
    "angle (nvidia, nvidia geforce gtx 980 direct3d11 vs_5_0 ps_5_0), or similar": "geforce gtx 980, or similar",
    "geforce gtx 980, or similar": "geforce gtx 980, or similar",
    "angle (amd, radeon r9 200 series direct3d11 vs_5_0 ps_5_0), or similar": "radeon r9 200 series, or similar",
    "radeon hd 3200 graphics, or similar": "radeon r9 200 series, or similar",
    "angle (intel, intel(r) hd graphics 400 direct3d11 vs_5_0 ps_5_0), or similar": "intel(r) hd graphics, or similar",
    "apple m1, or similar": "apple m1, or similar",
    "angle (nvidia geforce gtx 980) on vulkan, or similar": ", or similar",
    "generic renderer": ", or similar",
    "mesa intel(r) xe graphics (tgl gt2)": "iris(r) xe",
    "angle (intel, mesa intel(r) xe graphics (tgl gt2), opengl 4.6)": "iris(r) xe",
}


def first_match(s, entries):
    for e in entries:
        if e["match"] in s or any(a in s for a in e.get("aliases", ())):
            return e["match"]
    return None


def main():
    entries = [dict(zip(KEYS, row)) for row in G]
    problems = []
    seen = set()
    for e in entries:
        if e["match"] in ALIASES:
            e["aliases"] = ALIASES[e["match"]]
    for m in ALIASES:
        if m not in {e["match"] for e in entries}:
            problems.append(f"aliases for no entry: {m}")
    for e in entries:
        for text in [e["match"]] + e.get("aliases", []):
            if text != text.lower():
                problems.append(f"not lower-case: {text}")
            if text in seen:
                problems.append(f"duplicate match: {text}")
            seen.add(text)
        for a in e.get("aliases", []):
            got = first_match(a, entries)
            if got != e["match"]:
                problems.append(f"alias '{a}' of '{e['match']}' is taken by earlier '{got}'")
        if e["memory_gb"] is None and not e["notes"]:
            problems.append(f"null memory without a note: {e['match']}")
        if not e["source"].startswith("https://"):
            problems.append(f"bad source: {e['match']}")
        for t in templates(e["match"]):
            got = first_match(t, entries)
            if got != e["match"]:
                problems.append(f"'{e['match']}' is shadowed by earlier '{got}' in: {t}")
                break
    for s, want in {**REAL, **OTHER}.items():
        got = first_match(s, entries)
        if got != want:
            problems.append(f"real string mis-assigned: wanted '{want}', got '{got}': {s}")
    for p in problems:
        print("PROBLEM:", p)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "gpus.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1, ensure_ascii=False)
        f.write("\n")
    by_vendor = {}
    for e in entries:
        by_vendor[e["vendor"]] = by_vendor.get(e["vendor"], 0) + 1
    print(f"wrote {len(entries)} entries to {os.path.normpath(path)}; by vendor {by_vendor}; "
          f"null memory: {sum(e['memory_gb'] is None for e in entries)}; problems: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
