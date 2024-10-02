from functools import lru_cache
from typing import Optional, List

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap


torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def test_mask_one_task(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    fwdbwd: str = "fwd",
    repeats: int = 30,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
    sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = float('inf')
    sdpa_mask_time = float('inf')
    flex_ms = float('inf')
    if 'fwd' == fwdbwd:
        causal_fa2_time = do_bench(causal_fa2)
        sdpa_mask_time = do_bench(sdpa_mask)
        flex_ms = do_bench(flex_attention_call)
    
    causal_fa2_bw_time = float('inf')
    sdpa_mask_bw_time = float('inf')
    flex_bw_ms = float('inf')
    if 'bwd' == fwdbwd:
        # Backward pass
        causal_fa2_out = causal_fa2()
        sdpa_mask_out = sdpa_mask()
        flex_out = flex_attention_call()

        causal_fa2_bw_time = do_bench(lambda: causal_fa2_out.backward(gradOut, retain_graph=True))
        sdpa_mask_bw_time = do_bench(lambda: sdpa_mask_out.backward(gradOut, retain_graph=True))
        flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    # Inline correctness check
    if not skip_correctness:
        sdpa_mask_outs = []
        flex_outs = []

        for tensor in qkv:
            tensor.grad = None

        out1 = sdpa_mask()
        sdpa_mask_outs.append(out1)
        out1.backward(gradOut)
        sdpa_mask_outs += [tensor.grad for tensor in qkv]

        for tensor in qkv:
            tensor.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [tensor.grad for tensor in qkv]
        for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
            torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
        ],
        [
            "F.sdpa + mask",
            f"{sdpa_mask_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
            f"{sdpa_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
        ],
    ]
    return results
    

def test_mask_standard(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
    if mask_mod is not None:
        block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
    else:
        block_mask = None
    sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
    mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]
    gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv, is_causal=True)
    sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * B * H * D * S * S
    flops = density * B * H * D * S * S

    # Forward pass
    causal_fa2_time = do_bench(causal_fa2)
    sdpa_mask_time = do_bench(sdpa_mask)
    flex_ms = do_bench(flex_attention_call)

    # Backward pass
    causal_fa2_out = causal_fa2()
    sdpa_mask_out = sdpa_mask()
    flex_out = flex_attention_call()

    causal_fa2_bw_time = do_bench(lambda: causal_fa2_out.backward(gradOut, retain_graph=True))
    sdpa_mask_bw_time = do_bench(lambda: sdpa_mask_out.backward(gradOut, retain_graph=True))
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))

    print_header(
        f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
            "_", " "
        ).title()
    )
    # Inline correctness check
    if not skip_correctness:
        sdpa_mask_outs = []
        flex_outs = []

        for tensor in qkv:
            tensor.grad = None

        out1 = sdpa_mask()
        sdpa_mask_outs.append(out1)
        out1.backward(gradOut)
        sdpa_mask_outs += [tensor.grad for tensor in qkv]

        for tensor in qkv:
            tensor.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [tensor.grad for tensor in qkv]
        for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
            torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

        print("Correctness check passed ✅")
    # Usage in your results formatting:
    bhsd = [B, H, S, D]
    results = [
        [
            "causal FA2",
            f"{causal_fa2_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_time, 4):.2f}",
            f"{causal_fa2_bw_time:.4f}",
            f"{calculate_tflops(causal_fav2_flops, causal_fa2_bw_time, 10):.2f}",
            *bhsd,
        ],
        [
            "F.sdpa + mask",
            f"{sdpa_mask_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
            f"{sdpa_mask_bw_time:.4f}",
            f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
            *bhsd,
        ],
        [
            "flexattention",
            f"{flex_ms:.4f}",
            f"{calculate_tflops(flops, flex_ms, 4):.2f}",
            f"{flex_bw_ms:.4f}",
            f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
            *bhsd,
        ],
    ]
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time (ms)",
                "FW FLOPS (TF/s)",
                "BW Time (ms)",
                "BW FLOPS (TF/s)",
                "B",
                "H",
                "S",
                "D",
            ],
            tablefmt="grid",
        )
    )
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")


def test_mask(
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    batch_size=None,
    seqlen=None,
    headdim_vals=None,
    dim=None,
    repeats=30,
    fwdbwds=["fwd"],
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
):
    print_header(
        f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
            "_", " "
        ).title()
    )
    all_results = []
    printed_header = False
    for headdim in headdim_vals:
        for B in batch_size:
            for S in seqlen:
                for d in dim:
                    for fwdbwd in fwdbwds:
                        B, H, S, D = B, d // headdim, S, headdim
                        name = (score_mod.__name__ if score_mod is not None else mask_mod.__name__)
                        # print(f"Running {name} with B={B}, H={H}, S={S}, D={D}, fwdbwd={fwdbwd}")
                        result = test_mask_one_task(
                            score_mod=score_mod,
                            mask_mod=mask_mod,
                            B=B,
                            H=H,
                            S=S,
                            D=D,
                            fwdbwd=fwdbwd,
                            repeats=repeats,
                            skip_correctness=skip_correctness,
                            print_mask=print_mask,
                            device=device
                        )
                        for r in result:
                            r.extend([B, H, S, D, name])
                        all_results.extend(result)

                        headers=[
                            "Operation",
                            "FW Time (ms)",
                            "FW FLOPS (TF/s)",
                            "BW Time (ms)",
                            "BW FLOPS (TF/s)",
                            "B",
                            "H",
                            "S",
                            "D",
                            "Name",
                        ]
                        print(
                            tabulate(
                                result,
                                headers=headers,
                                tablefmt="grid",
                            )
                        )
    return all_results

def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768)


def main_standard(
    examples: List[str] = ["all"], 
    print_mask = False
):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    # B: int = 16,
    # H: int = 16,
    # S: int = 8192,
    # D: int = 64,
    
    available_examples = {
        "causal": lambda: test_mask(mask_mod=causal_mask, print_mask=print_mask),
        "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), skip_correctness=True, print_mask=print_mask),
        "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024), print_mask=print_mask),
        "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024), print_mask=print_mask),
        "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12, print_mask=print_mask),
        "softcap": lambda: test_mask(
            score_mod=generate_tanh_softcap(30, approx=False), skip_correctness=True, print_mask=print_mask
        ),
        "softcap_approx": lambda: test_mask(
            score_mod=generate_tanh_softcap(30, approx=True), skip_correctness=True, print_mask=print_mask
        ),
    }

    if "all" in examples:
        ex_to_run = list(available_examples.keys())
    else:
        ex_to_run = examples

    all_results = []
    for ex in ex_to_run:
        if ex in available_examples:
            result = available_examples[ex]()
            all_results.extend(result)
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")
    
    headers=[
        "Operation",
        "FW Time (ms)",
        "FW FLOPS (TF/s)",
        "BW Time (ms)",
        "BW FLOPS (TF/s)",
        "B",
        "H",
        "S",
        "D",
        "Name",
    ]
    # export to csv
    with open("flexattn_benchmark.csv", "w") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_results)

def main(
    examples: List[str] = ["all"], 
    print_mask = False,
    # bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 8192 * 2)],
    batch_size = [2],
    seqlen = [1024, 2048, 4096, 8192, 16384, 32768],
    headdim_vals = [64, 128, 256],
    dim = [2048],
    repeats = 30,
    fwdbwds = ['fwd'],
    skip_correctness = False,
):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    # B: int = 16,
    # H: int = 16,
    # S: int = 8192,
    # D: int = 64,
    kwargs = dict(
        batch_size=batch_size,
        seqlen=seqlen,
        headdim_vals=headdim_vals,
        dim=dim,
        repeats=repeats,
        fwdbwds=fwdbwds,
        skip_correctness=skip_correctness,
        print_mask=print_mask,
    )
    
    available_examples = {
        "causal": lambda: test_mask(mask_mod=causal_mask, **kwargs),
        "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), **kwargs),
        "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024), **kwargs),
        "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024), **kwargs),
        "softcap": lambda: test_mask(
            score_mod=generate_tanh_softcap(30, approx=False), **kwargs
        ),
        "softcap_approx": lambda: test_mask(
            score_mod=generate_tanh_softcap(30, approx=True), **kwargs
        ),

        # TODO: Need to change interface
        # "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12, **kwargs),
    }

    if "all" in examples:
        ex_to_run = list(available_examples.keys())
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in available_examples:
            available_examples[ex]()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")



if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: causal, alibi, sliding_window, prefix_lm, "
        "document, softcap, softcap_approx, or 'all' to run all examples.",
    )
    parser.add_argument(
        "--print_mask",
        action="store_true",
        help="Print the mask matrix.",
    )
    parser.add_argument(
        "--skip_correctness",
        action="store_true",
        help="Skip correctness check.",
    )

    args = parser.parse_args()
    main(**vars(args))
