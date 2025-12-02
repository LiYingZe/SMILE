import argparse
import time
import torch
from torch.nn.utils.rnn import pad_sequence

from SLM_LIKE import (
    OptimizedTransformerSeq2Seq,
    CharacterLevelDataset,
    generateListOfWildCards,
    load_data,
    split_list,
    optimized_test,
    setup_seed,
)


def build_model_and_dataset(args, device):
    tuples = load_data(args.data_path, args.numTem)
    if not tuples:
        raise RuntimeError("No data loaded from data_path.")

    X_test = tuples[: args.eval_size]
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

    return model, dataset, X_test, Y_test


def evaluate_recall(model, dataset, X_test, Y_test, args, device):
    sqt = [[Y_test[i], [X_test[i]]] for i in range(len(X_test))]
    recall = 0.0
    batches = split_list(sqt, args.inferParrellism)

    for inference_samples in batches:
        input_texts = []
        labels = []
        queries = []
        for wi, res_list in inference_samples:
            queries.append(wi)
            labels.append(res_list)
            input_indices = (
                [dataset.char_to_idx["<SOS>"]]
                + [dataset.char_to_idx[c] for c in wi]
                + [dataset.char_to_idx["<EOS>"]]
            )
            input_texts.append(torch.tensor(input_indices))

        input_texts_padded = pad_sequence(
            input_texts,
            padding_value=dataset.char_to_idx["<PAD>"],
            batch_first=True,
        ).to(device)

        t0 = time.time()
        beam_results = optimized_test(
            input_texts_padded,
            model,
            dataset,
            device,
            K=int(args.inferSampleNum),
            temperature=0.9,
        )
        t1 = time.time()
        print(
            f"Batched inference time: {t1 - t0:.2f}s, per query: {(t1 - t0) / len(inference_samples):.4f}s"
        )

        predict_per_query = split_list(beam_results, k=int(args.inferSampleNum))

        for pred_idx, (candidates, ground_truth) in enumerate(
            zip(predict_per_query, labels)
        ):
            hit_count = sum(1 for gt in ground_truth if gt in candidates)
            recall += (hit_count / len(ground_truth)) if ground_truth else 0
            if pred_idx < 2:
                print(
                    f"Sample | Wildcard: {queries[pred_idx]}, GT: {ground_truth}, "
                    f"Recall: {hit_count}/{len(ground_truth)}, Pred: {candidates[:3]}..."
                )

    total = len(sqt)
    print(f"Total Recall Sum: {recall:.4f}, Average Recall: {recall / total:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate seq2seq model (optimized).")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/lineitem10000.csv",
        help="Path to data file used to rebuild vocab.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model state_dict (torch.load).",
    )
    parser.add_argument("--HIDDEN_SIZE", type=int, default=512)
    parser.add_argument("--LayerNum", type=int, default=4)
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--inferSampleNum", type=int, default=4)
    parser.add_argument("--inferParrellism", type=int, default=64)
    parser.add_argument("--pct", type=float, default=0.2)
    parser.add_argument("--inPct", type=float, default=0.1)
    parser.add_argument("--numTem", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_size", type=int, default=256)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.GPU}" if torch.cuda.is_available() else "cpu")

    model, dataset, X_test, Y_test = build_model_and_dataset(args, device)
    evaluate_recall(model, dataset, X_test, Y_test, args, device)


if __name__ == "__main__":
    main()