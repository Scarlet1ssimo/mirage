# Artifact Evaluation of Mirage

## Getting Started Instructions
You may install the baseline methods manually by following their respective official tutorials. Alternatively, if you are using the machine we provide, you can directly use the pre-configured Conda environment, which includes all dependencies required for Mirage and the baseline implementations.
```
conda activate mirage_ae
```
Clone the Mirage source code from github and go to the branch for artifact evaluation.
```
git clone https://github.com/mirage-project/mirage.git --recursive
cd mirage
git checkout evaluation
```
Install Mirage from source
```
export MIRAGE_ROOT=/path/to/mirage
cd $MIRAGE_ROOT
pip install .
```

## Experiment Instructions

You can run
```
python3 $MIRAGE_ROOT/ae_scripts/run_all.py
```
to generate all the results in Figure 7 and Figure 11.

For the results on Grace Hopper GPU(H100), you can run

```
python3 $MIRAGE_ROOT/ae_scripts/run_all_hopper.py
```
To generate the search time study results in Table 5, run

```
python3 $MIRAGE_ROOT/ae_scripts/search_time.py
```


You can also generate each data point separately. Followings are the detailed instructions.

### Micro-benchmark Evaluation
Baseline implementations are in `$MIRAGE_ROOT/mirage_baselines`.

To reproduce the benchmark results shown in the paper, run the python scripts under `$MIRAGE_ROOT/benchmark`. The optimization time may take up to 4 hours. In order to save time, you can use our cached results in `$MIRAGE_ROOT/benchmark/saved_mugraphs` to avoid running from scratch.

To evaluate the optimization time, run Mirage from scratch, and look at the line
```
[Search] Second step finished. Time elapsed: xx.xxsec
```
which reports the optimization time.

Specifically, followings are how to reproduce each benchmark result:


#### GQA
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gqa_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/group_query_attention.py -b {batch_size}
```


#### QKNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/qknorm_gqa.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/qknorm_gqa_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/qknorm_gqa.py -b {batch_size}
```

#### RMSNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/rmsnorm_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py -b {batch_size}
```

#### RMSNorm:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/rmsnorm_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/rmsnorm.py -b {batch_size}
```

#### LoRA:
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/lora.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/lora_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/lora.py -b {batch_size}
```

#### MHA
Use cached data with the CUDA backend:
``` 
python3 $MIRAGE_ROOT/benchmark/multi_head_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/mha_bs{batch_size}.json --backend cuda
```

To replay the CUDA path as a standalone script with configurable shape, run:
``` 
python3 $MIRAGE_ROOT/benchmark/multi_head_attention_cuda_replay.py --batch_size 1 --outer_dim 32 --q_len 16 --k_len 4096 --d_model 64 --file $MIRAGE_ROOT/benchmark/saved_mugraphs/mha_bs1.json
```
If you change the shape, also pass a matching checkpoint if you have one; otherwise omit `--file` to let Mirage search for the new shape.

You can tune CUDA schedule knobs directly from the standalone script:
``` 
python3 $MIRAGE_ROOT/benchmark/multi_head_attention_cuda_replay.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/mha_bs1.json --pipeline_stages 2 --num_warp_groups 2 --target_cc 90 --file_id 0
```
The generated CUDA source is exported to `$MIRAGE_ROOT/generated_codes/{file_id}/test{file_id}.cu`.

To extract Triton code from the cached search results, run:
``` 
python3 $MIRAGE_ROOT/benchmark/multi_head_attention.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/mha_bs1.json --backend triton --save_codes True
```
The generated Triton file is written to `$MIRAGE_ROOT/generated_codes/generated_code_0.py`.

To replay the extracted Triton kernel and measure latency without rerunning search, run:
``` 
python3 $MIRAGE_ROOT/benchmark/multi_head_attention_triton_replay.py
```

#### GatedMLP
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/gated_mlp.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/gated_mlp_bs{batch_size}.json
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/gated_mlp.py -b {batch_size}
```

#### nTrans
Use cached data:
```
python3 $MIRAGE_ROOT/benchmark/norm_transformer.py --file $MIRAGE_ROOT/benchmark/saved_mugraphs/ntrans_bs{batch_size}.py
```
From scratch:
```
python3 $MIRAGE_ROOT/benchmark/norm_transformer.py -b {batch_size}
```

### End-to-end Evaluation
Baseline implementations are in `$MIRAGE_ROOT/demo/pytorch`.

To reproduce the end-to-end evaluation results shown in the paper, run the python scripts under `$MIRAGE_ROOT/benchmark/end-to-end`. Optimization usually takes less than 1 min.

#### Chameleon-7B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/chameleon.py -b {batch_size}
```

#### LLaMA-3-8B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/llama.py -b {batch_size}
```

#### GPT-3-7B-LoRA
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/lora.py -b {batch_size}
```

#### nGPT-1B
```
python3 $MIRAGE_ROOT/benchmark/end-to-end/ngpt.py -b {batch_size}
```
