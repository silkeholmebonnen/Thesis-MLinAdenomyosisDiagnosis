import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path

def plot_and_save(y, job_id, musa, image_id, title):
    plt.figure() 
    idx = range(0,200)
    plt.plot(idx, y)
    plt.title(title)
    plt.xlabel("Slices")
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    outdir = Path(f"reports/{job_id}/{musa}/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/pred_prob_per_slice_for_img{image_id}.png")
    plt.close()

def plot_and_save_with_std(y, y_std, job_id, musa, image_id, title):
    plt.figure() 
    idx = range(0,200)
    plt.plot(idx, y, label="Mean")
    plt.fill_between(
        idx,
        y - y_std,
        y + y_std,
        alpha=0.3,
        label="±1 std"
    )
    plt.title(title)
    plt.xlabel("Slices")
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.legend()
    outdir = Path(f"reports/{job_id}/{musa}/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/pred_prob_per_slice_for_img{image_id}.png")
    plt.close()

def main():
    job_id = "840578"
    musa = "irregular jz"
    path = f"reports/{job_id}/{musa}/validation/posterier/fold4.csv"
    df = pd.read_csv(path)
    unique_img_id = df["image_ids"].unique()

    for image_id in unique_img_id:
        image_df = df[df["image_ids"] == image_id]
        title = f"Predicted prob over the 200 slices. Correct label={image_df['true'].iloc[0]}"
        plot_and_save(image_df["predicted prob"], job_id, musa, image_id, title)
        

if __name__=="__main__":
    main()