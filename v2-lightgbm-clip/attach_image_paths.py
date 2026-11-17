import os, os.path as osp, argparse, pandas as pd

def find_path(root, sample_id):
    sample_id = str(int(sample_id))
    for ext in ('.jpg','.jpeg','.png','.webp'):
        p = osp.join(root, sample_id+ext)
        if osp.exists(p):
            return p
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--images_root", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv_in)
    paths = [find_path(args.images_root, sid) for sid in df["sample_id"]]
    df["image_path"] = paths
    df.to_csv(args.csv_out, index=False)
    print("Wrote", args.csv_out)
