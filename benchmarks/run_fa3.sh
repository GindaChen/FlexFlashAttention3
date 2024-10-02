mkdir -p results
SCRIPT_DIR=$(dirname $0)
FLASH_ATTENTION_DIR=$SCRIPT_DIR/../flash-attention
# check existence of FLASH_ATTENTION_DIR
if [ ! -d "$FLASH_ATTENTION_DIR" ]; then
    echo "FLASH_ATTENTION_DIR does not exist: $FLASH_ATTENTION_DIR"
    exit 1
fi
export PYTHONPATH=$PYTHONPATH:$FLASH_ATTENTION_DIR/hopper
python fa3_benchmark.py > results/fa3_benchmark.log
