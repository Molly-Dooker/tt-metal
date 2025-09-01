##### build
```
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh   --release
./create_venv.sh
source python_env/bin/activate
pip install pyarrow==13.0.0
pip install -r models/tt_transformers/requirements.txt
```

##### resnet50 benchmark
```
# prepare weigth tensor
python models/demos/bos_model/resnet50/_prepare_weight.py

# run default model
python models/demos/bos_model/resnet50/main.py --trace --modelpath 'models/demos/bos_model/resnet50/fp32.bin'

# run int8 quantized model
python models/demos/bos_model/resnet50/main.py --trace --modelpath 'models/demos/bos_model/resnet50/int8.bin'
```


##### llama sample
```
export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export MESH_DEVICE=N150

# batch 1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# batch 32
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"
```
