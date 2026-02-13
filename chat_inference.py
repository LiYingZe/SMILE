import argparse
import time
import torch
from torch.nn.utils.rnn import pad_sequence

from SLM_LIKE import (
    OptimizedTransformerSeq2Seq,
    CharacterLevelDataset,
    generateListOfWildCards,
    load_data,
    setup_seed,
)


def build_model_and_dataset(args, device):
    tuples = load_data(args.data_path, args.numTem)
    if not tuples:
        raise RuntimeError("No data loaded from data_path.")

    X_test = tuples[: args.vocab_seed_size]
    Y_test = generateListOfWildCards(
        X_test,
        args.inPct,
        wildcard_prob={"%": args.pct, "_": 1 - args.pct},
    )
    dataset = CharacterLevelDataset(X_test + tuples, Y_test)

    model = OptimizedTransformerSeq2Seq(
        src_vocab_size=dataset.src_vocab_size,
        trg_vocab_size=dataset.trg_vocab_size,
        d_model=args.HIDDEN_SIZE,
        nhead=8,
        num_layers=args.LayerNum,
        dim_feedforward=args.HIDDEN_SIZE,
        max_seq_length=1024,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, dataset


def interactive_loop(model, dataset, args, device):
    print("\nModel loaded. Enter a LIKE pattern (e.g., a%cd_). Type 'exit' or 'q' to quit.\n")
    while True:
        like_pattern = input("LIKE pattern > ").strip()
        if like_pattern.lower() in {"exit", "q"}:
            print("Exiting.")
            break
        if not like_pattern:
            print("Input cannot be empty. Please enter a valid pattern.")
            continue

        try:
            indices = (
                [dataset.char_to_idx["<SOS>"]]
                + [dataset.char_to_idx[c] for c in like_pattern]
                + [dataset.char_to_idx["<EOS>"]]
            )
            input_tensor = pad_sequence(
                [torch.tensor(indices)],
                padding_value=dataset.char_to_idx["<PAD>"],
                batch_first=True,
            ).to(device)

            with torch.no_grad():
                t0 = time.time()
                outputs = model(
                    input_tensor,
                    torch.full(
                        (input_tensor.size(0), 1),
                        dataset.char_to_idx["<SOS>"],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                # Use optimized_test-style sampling for multiple candidates
                from SLM_LIKE import optimized_test

                results = optimized_test(
                    input_tensor,
                    model,
                    dataset,
                    device,
                    K=int(args.inferSampleNum),
                    temperature=0.9,
                )
                t1 = time.time()

            topk = results[: int(args.inferSampleNum)]
            print(f"\nResults (Top {args.inferSampleNum}):")
            for i, res in enumerate(topk):
                print(f"{i + 1}. {res}")
            print(f"Time: {t1 - t0:.2f} s\n")

        except KeyError:
            print("Input contains characters not in the vocabulary built from data_path.")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive inference with optimized seq2seq.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/lineitem.csv",
        help="Path to data file used to rebuild vocab.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./lineitem/best_model.pth",
        help="Path to model state_dict (torch.load).",
    )
    parser.add_argument("--HIDDEN_SIZE", type=int, default=512)
    parser.add_argument("--LayerNum", type=int, default=4)
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--inferSampleNum", type=int, default=4)
    parser.add_argument("--pct", type=float, default=0.2)
    parser.add_argument("--inPct", type=float, default=0.1)
    parser.add_argument("--numTem", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_seed_size", type=int, default=256, help="Number of examples to build vocab.")
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")

    model, dataset = build_model_and_dataset(args, device)
    interactive_loop(model, dataset, args, device)


if __name__ == "__main__":
    main()


