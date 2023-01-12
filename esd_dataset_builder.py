import os


def run(wav_path, manifests_path, target_path):
    parts = os.listdir(manifests_path)
    for part in parts:
        if ".txt" not in part:
            continue
        lines = open(os.path.join(manifests_path, part)).readlines()
        if part == "test.txt":
            target_manifest_path = os.path.join(target_path, "test.txt")
        else:
            target_manifest_path = os.path.join(target_path, "train.txt")

        for line in lines:
            speaker, idx, emo, _, _ = line.split("|")
            wav_name = f"{speaker}_{idx}_{emo}.wav"
            with open(target_manifest_path, "a") as f:
                f.write(os.path.join(wav_path, wav_name))
                f.write("\n")


if __name__ == "__main__":
    wav_path = "/root/storage/dasha/data/emo-data/esd/punc_768/trimmed_wav"
    manifests_path = "/root/storage/dasha/data/emo-data/esd/punc_768"
    target_path = "/root/storage/dasha/data/emo-data/esd/istft"
    run(wav_path, manifests_path, target_path)
