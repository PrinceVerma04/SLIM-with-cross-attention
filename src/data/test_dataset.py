from src.data.audio_dataset import AudioDataset


def main():
    dataset = AudioDataset("data/splits/stage2_train.csv", duration_sec=2.0)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print("\nSample keys:", sample.keys())
    print("Waveform shape:", sample["waveform"].shape)
    print("Label:", sample["label"])
    print("Filepath:", sample["filepath"])


if __name__ == "__main__":
    main()