import argparse
import torch

from src.utils.audio_utils import preprocess_audio
from src.models.stage2_slim import Stage2SLIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for FAKE class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = Stage2SLIM().to(device)
    model.load_state_dict(torch.load("outputs/stage2_best.pt", map_location=device))
    model.eval()

    # Load and preprocess audio
    waveform = preprocess_audio(args.audio, duration_sec=5.0)   # [1, T]
    waveform = waveform.unsqueeze(0).to(device)                 # [1, 1, T]

    with torch.no_grad():
        outputs = model(waveform)
        logits = outputs["logits"]
        prob_fake = torch.sigmoid(logits)[0].item()
        prob_real = 1.0 - prob_fake

        pred_class = int(prob_fake >= args.threshold)
        confidence = prob_fake if pred_class == 1 else prob_real

    label_map = {
        0: "REAL",
        1: "FAKE"
    }

    print("\n=== Prediction Result ===")
    print(f"Audio:       {args.audio}")
    print(f"Prediction:  {label_map[pred_class]}")
    print(f"Confidence:  {confidence:.4f}")
    print(f"Threshold:   {args.threshold:.4f}")
    print(f"Prob REAL:   {prob_real:.4f}")
    print(f"Prob FAKE:   {prob_fake:.4f}")


if __name__ == "__main__":
    main()
