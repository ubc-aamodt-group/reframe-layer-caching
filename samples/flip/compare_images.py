import flip_evaluator
import os

scores = []

baseline_dir = "baseline_results"
reframe_dir = "reframe_results"
scene_name = "SunTemple"

baseline_images = [f for f in os.listdir(baseline_dir) if scene_name in f and ".png" in f]
baseline_images.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
reframe_images = [f for f in os.listdir(reframe_dir) if scene_name in f and ".png" in f]
reframe_images.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))


for r, t in zip(baseline_images, reframe_images):
    frame = r.split(".")[-2]
    print(f"Frame: {frame}", end="\r")
    img, score, _ = flip_evaluator.evaluate(
        f"{baseline_dir}/{r}",
        f"{reframe_dir}/{t}",
        "LDR")
    scores.append(score)

print(f"Average FLIP Score: {sum(scores)/len(scores):.3f}")

output_csv = "flip_scores.csv"

with open(output_csv, "w+") as f:
    f.write("Frame,PNG Score\n")

    # Add scores
    for i, score in enumerate(scores):
        f.write(f"{i+1},{score:.3f}\n")

print(f"Scores written to {output_csv}")

