"""
Build data/splits CSVs for SLIM-CSAI using:
  - ASVspoof 2019 LA  (train / dev / eval)
  - MILABS (m_ailabs_mlaad)  — extra real speech for Stage 1 diversity

Only 50% of each dataset split is used (class-balanced sampling) to
reduce training time while preserving real/fake ratio.

Stage 1 CSV  (real only, self-supervised contrastive training):
  50% of ASVspoof 2019 LA train bonafide + 50% of MILABS train real

Stage 2 CSVs (real + fake, supervised):
  train → 50% of ASVspoof 2019 LA train  (stratified by label)
  val   → 50% of ASVspoof 2019 LA dev    (stratified by label)
  test  → 50% of ASVspoof 2019 LA eval   (stratified by label)
"""

import csv
import random
from pathlib import Path

random.seed(42)
SAMPLE_FRAC = 0.5   # use 50% of each split

BASE   = Path(__file__).resolve().parents[2]
ASV_LA = BASE / "AVSSPOOF_2019/ASVSpoof_2019/LA/LA"
MILABS = BASE / "MILABS/m_ailabs_mlaad"
SPLITS = BASE / "data/splits"
SPLITS.mkdir(parents=True, exist_ok=True)


def parse_asv_protocol(proto_path, flac_dir):
    """
    Parse an ASVspoof 2019 LA CM protocol file.
    Format: speaker_id  file_id  -  -  bonafide/spoof
    Returns list of (filepath, label, speaker_id).
    label: 0=real (bonafide), 1=fake (spoof).
    Silently skips files missing on disk.
    """
    rows = []
    with open(proto_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            spk_id, file_id, verdict = parts[0], parts[1], parts[4]
            label = 0 if verdict == "bonafide" else 1
            flac_path = flac_dir / f"{file_id}.flac"
            if flac_path.exists():
                rows.append((str(flac_path), label, spk_id))
    return rows


def sample_balanced(rows, frac=0.5):
    """
    Sample `frac` of rows while preserving real/fake class balance.
    Each class is sampled independently then merged and shuffled.
    """
    real = [r for r in rows if r[1] == 0]
    fake = [r for r in rows if r[1] == 1]

    random.shuffle(real)
    random.shuffle(fake)

    real_sampled = real[:max(1, int(len(real) * frac))]
    fake_sampled = fake[:max(1, int(len(fake) * frac))]

    combined = real_sampled + fake_sampled
    random.shuffle(combined)
    return combined


def write_csv(rows, path, header=("filepath", "label", "speaker_id")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    real_n = sum(1 for r in rows if r[1] == 0)
    fake_n = sum(1 for r in rows if r[1] == 1)
    print(f"  {path.name:<25}  total={len(rows):>6}  real={real_n:>6}  fake={fake_n:>6}")


# ── Stage 1: real-only (50% of bonafide) ─────────────────────────────────────

print(f"Building Stage 1 split (real only, {int(SAMPLE_FRAC*100)}% sample) …")

asv_train_proto = ASV_LA / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
asv_train_flac  = ASV_LA / "ASVspoof2019_LA_train/flac"
asv_train_all   = parse_asv_protocol(asv_train_proto, asv_train_flac)
asv_train_real  = [r for r in asv_train_all if r[1] == 0]
asv_train_real_sampled = asv_train_real[:max(1, int(len(asv_train_real) * SAMPLE_FRAC))]

milabs_real = []
milabs_real_dir = MILABS / "train/real"
if milabs_real_dir.exists():
    all_milabs = [(str(f), 0, "milabs") for f in sorted(milabs_real_dir.glob("*.wav"))]
    milabs_real = all_milabs[:max(1, int(len(all_milabs) * SAMPLE_FRAC))]

stage1_rows = asv_train_real_sampled + milabs_real
random.shuffle(stage1_rows)
write_csv(stage1_rows, SPLITS / "stage1_real.csv")


# ── Stage 2: train (50% of ASVspoof 2019 LA train, balanced) ─────────────────

print(f"\nBuilding Stage 2 splits ({int(SAMPLE_FRAC*100)}% sample, class-balanced) …")
write_csv(sample_balanced(asv_train_all, SAMPLE_FRAC), SPLITS / "stage2_train.csv")


# ── Stage 2: val (50% of ASVspoof 2019 LA dev, balanced) ─────────────────────

asv_dev_proto = ASV_LA / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
asv_dev_flac  = ASV_LA / "ASVspoof2019_LA_dev/flac"
asv_dev_all   = parse_asv_protocol(asv_dev_proto, asv_dev_flac)
write_csv(sample_balanced(asv_dev_all, SAMPLE_FRAC), SPLITS / "stage2_val.csv")


# ── Stage 2: test (50% of ASVspoof 2019 LA eval, balanced) ───────────────────

asv_eval_proto = ASV_LA / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
asv_eval_flac  = ASV_LA / "ASVspoof2019_LA_eval/flac"
asv_eval_all   = parse_asv_protocol(asv_eval_proto, asv_eval_flac)
write_csv(sample_balanced(asv_eval_all, SAMPLE_FRAC), SPLITS / "stage2_test.csv")

print("\nAll splits written to data/splits/")


if __name__ == "__main__":
    pass
