"""Laptop vs desktop table for the Laptop Tutorial Benchmark: reads C:/v4runs/laptop/runs/*/report.json and the
desktop's numbers (copied from the hand-off page), writes laptop_vs_desktop.csv and .md next to the runs."""
import csv, json, sys
from pathlib import Path

BASE = json.loads(r'''[{"n":"t_CT_GEN3","s":289,"ram":17.1,"gpu":1.63,"lo":9.7,"hi":17.0,"r":1},{"n":"t_CT_microct","s":52,"ram":14.9,"gpu":6.66,"lo":1.8,"hi":3.2,"r":1},{"n":"t_CardiacReorientation","s":4,"ram":6.2,"gpu":0.06,"lo":0.2,"hi":0.3,"r":1},{"n":"t_GE_HDF5","s":145,"ram":11.4,"gpu":1.8,"lo":4.3,"hi":7.5,"r":1},{"n":"t_PETGATE_DIP","s":867,"ram":22.2,"gpu":8.02,"lo":44.6,"hi":76.8,"r":2},{"n":"t_PETGATE_scat_lm","s":89,"ram":21.9,"gpu":3.22,"lo":6.5,"hi":11.0,"r":2},{"n":"t_PETGATE_scat_lmTOF","s":100,"ram":22.6,"gpu":3.9,"lo":7.2,"hi":12.3,"r":2},{"n":"t_PETGATE_scat_sino","s":573,"ram":20.2,"gpu":1.53,"lo":17.0,"hi":29.9,"r":2},{"n":"t_PETGATE_scat_sinoTOF","s":661,"ram":20.7,"gpu":5.61,"lo":12.4,"hi":22.6,"r":2},{"n":"t_PETSIRD","s":145,"ram":25.2,"gpu":5.98,"lo":8.3,"hi":14.2,"r":1},{"n":"t_ac225_dicom_recon","s":127,"ram":14.4,"gpu":2.98,"lo":9.6,"hi":16.4,"r":2},{"n":"t_ac225_dualpeak","s":307,"ram":11.4,"gpu":3.04,"lo":23.9,"hi":40.7,"r":2},{"n":"t_ac225_simind_recon","s":96,"ram":14.0,"gpu":3.08,"lo":7.4,"hi":12.6,"r":1},{"n":"t_accuracy_known_truth","s":17,"ram":6.9,"gpu":0.42,"lo":1.2,"hi":2.0,"r":1},{"n":"t_algorithms","s":41,"ram":9.0,"gpu":1.58,"lo":3.0,"hi":5.0,"r":1},{"n":"t_algorithms_measured","s":93,"ram":9.4,"gpu":1.69,"lo":7.1,"hi":12.2,"r":1},{"n":"t_arf_psf","s":151,"ram":10.5,"gpu":1.98,"lo":11.4,"hi":19.3,"r":2},{"n":"t_cgls","s":132,"ram":9.4,"gpu":0.75,"lo":10.0,"hi":17.0,"r":2},{"n":"t_dicomdata","s":7,"ram":6.8,"gpu":0.39,"lo":0.3,"hi":0.5,"r":1},{"n":"t_dicommultibed","s":18,"ram":8.1,"gpu":0.44,"lo":0.7,"hi":1.3,"r":1},{"n":"t_dualpeak","s":7,"ram":6.9,"gpu":0.47,"lo":0.4,"hi":0.7,"r":1},{"n":"t_examplesystemmatrix","s":4,"ram":4.4,"gpu":0.0,"lo":0.3,"hi":0.5,"r":1},{"n":"t_fbp_ct","s":109,"ram":14.5,"gpu":1.3,"lo":3.6,"hi":6.2,"r":1},{"n":"t_fbp_spect","s":6,"ram":7.1,"gpu":0.43,"lo":0.4,"hi":0.6,"r":1},{"n":"t_kernel_deep_kernel_spect_recon","s":195,"ram":11.7,"gpu":1.44,"lo":14.3,"hi":24.4,"r":2},{"n":"t_pct_abdomen","s":49,"ram":9.2,"gpu":0.82,"lo":2.9,"hi":5.0,"r":1},{"n":"t_pet_introduction","s":374,"ram":18.8,"gpu":2.98,"lo":11.4,"hi":20.0,"r":1},{"n":"t_plotting","s":10,"ram":5.8,"gpu":0.0,"lo":0.2,"hi":0.4,"r":1},{"n":"t_siminddata","s":5,"ram":6.8,"gpu":0.39,"lo":0.3,"hi":0.4,"r":1},{"n":"t_spect_mc","s":2458,"ram":7.0,"gpu":0.46,"lo":53.9,"hi":103.5,"r":2},{"n":"t_spect_mc2","s":2198,"ram":7.1,"gpu":0.46,"lo":48.3,"hi":92.7,"r":2},{"n":"t_starguide","s":16,"ram":7.9,"gpu":1.1,"lo":0.9,"hi":1.5,"r":1},{"n":"t_uncertainty_pet","s":118,"ram":13.1,"gpu":2.07,"lo":2.0,"hi":3.7,"r":1},{"n":"t_uncertainty_spect","s":18,"ram":10.2,"gpu":0.71,"lo":1.0,"hi":1.6,"r":1}]''')
ORDER = ("t_siminddata t_dicomdata t_algorithms t_accuracy_known_truth t_dicommultibed t_dualpeak t_uncertainty_spect "
         "t_starguide t_ac225_simind_recon t_CardiacReorientation t_algorithms_measured t_fbp_spect t_arf_psf "
         "t_ac225_dicom_recon t_ac225_dualpeak t_cgls t_kernel_deep_kernel_spect_recon t_examplesystemmatrix t_plotting "
         "t_CT_GEN3 t_CT_microct t_fbp_ct t_pct_abdomen t_pet_introduction t_GE_HDF5 t_uncertainty_pet t_PETSIRD "
         "t_PETGATE_scat_sino t_PETGATE_scat_sinoTOF t_PETGATE_scat_lm t_PETGATE_scat_lmTOF t_PETGATE_DIP "
         "t_spect_mc t_spect_mc2").split()
MOD = {**{n: "SPECT" for n in ORDER[:17] + ["t_spect_mc", "t_spect_mc2"]}, "t_examplesystemmatrix": "Any",
       "t_plotting": "Any", "t_CT_GEN3": "CT", "t_CT_microct": "CT", "t_fbp_ct": "CT", "t_pct_abdomen": "Proton CT",
       **{n: "PET" for n in ORDER[23:32]}}

root = Path(sys.argv[1])
base = {b["n"]: b for b in BASE}
rows = []
for n in ORDER:
    b, rep = base[n], root / "runs" / n / "report.json"
    r = json.loads(rep.read_text(encoding="utf8"))[0] if rep.exists() else None
    info = (r or {}).get("info", {})
    t = r["wall_time_s"] if r else None
    rows.append(dict(tutorial=n, modality=MOD[n], desktop_round=b["r"],
                     desktop_s=b["s"], desktop_ram_gb=b["ram"], desktop_gpu_gb=b["gpu"],
                     estimate_min=f"{b['lo']}-{b['hi']}",
                     laptop_status=r["status"] if r else "not run", laptop_s=t,
                     laptop_ram_gb=r.get("peak_kernel_gb") if r else None, laptop_gpu_gb=info.get("peak_gpu_gb"),
                     slower=round(t / b["s"], 2) if r and r["status"] == "passed" else None))

with open(root / "laptop_vs_desktop.csv", "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

ran = [x for x in rows if x["laptop_s"] is not None]
passed = [x for x in ran if x["laptop_status"] == "passed"]
lap_total = sum(x["laptop_s"] for x in ran)
desk_same = sum(x["desktop_s"] for x in ran)
desk_passed = sum(x["desktop_s"] for x in passed)
lap_passed = sum(x["laptop_s"] for x in passed)
fmt = lambda s: f"{s/60:.1f} min" if s >= 90 else f"{s:.0f} s"
md = ["| Tutorial | Modality | Desktop | Laptop | Slower | Laptop RAM / GPU (GB) | Desktop RAM / GPU (GB) | Laptop result |",
      "|---|---|---|---|---|---|---|---|"]
for x in rows:
    md.append(f"| {x['tutorial']} | {x['modality']} | {fmt(x['desktop_s'])} | {fmt(x['laptop_s']) if x['laptop_s'] is not None else ''} | "
              f"{str(x['slower']) + '×' if x['slower'] else ''} | {x['laptop_ram_gb'] if x['laptop_ram_gb'] is not None else ''} / {x['laptop_gpu_gb'] if x['laptop_gpu_gb'] is not None else ''} | "
              f"{x['desktop_ram_gb']} / {x['desktop_gpu_gb']} | {x['laptop_status']} |")
summary = [f"Ran {len(ran)} of {len(rows)}; passed {len(passed)}.",
           f"Laptop total {lap_total/60:.1f} min for the {len(ran)} run (desktop {desk_same/60:.1f} min for the same set); "
           f"passed only: laptop {lap_passed/60:.1f} min vs desktop {desk_passed/60:.1f} min = {lap_passed/desk_passed:.2f}x.",
           "Biggest slowdowns: " + ", ".join(f"{x['tutorial']} {x['slower']}x" for x in sorted(passed, key=lambda x: -x['slower'])[:5]),
           "Smallest: " + ", ".join(f"{x['tutorial']} {x['slower']}x" for x in sorted(passed, key=lambda x: x['slower'])[:5])]
for mod in ["SPECT", "Any", "CT", "Proton CT", "PET"]:
    g = [x for x in passed if x["modality"] == mod]
    if g:
        summary.append(f"{mod}: {len(g)} passed, laptop {sum(x['laptop_s'] for x in g)/60:.1f} min vs desktop "
                       f"{sum(x['desktop_s'] for x in g)/60:.1f} min ({sum(x['laptop_s'] for x in g)/sum(x['desktop_s'] for x in g):.2f}x)")
(root / "laptop_vs_desktop.md").write_text("\n".join(summary) + "\n\n" + "\n".join(md) + "\n", encoding="utf8")
print("\n".join(summary))
