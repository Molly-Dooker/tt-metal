import argparse
import sys

import evaluate
import torch
import torchvision
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

import ttnn
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.utility_functions import profiler

BATCH_SIZE = 16
CACHE_DIR = "/home/workspace/Dataset/ImageNet"
NUM_WORKERS = 4


def transform(data_batch, processor):
    inputs = {
        "pixel_values": torch.stack([processor(image.convert("RGB")) for image in data_batch["image"]]),
        "labels": data_batch["label"],
    }
    return inputs


def logger_setup(prefix: str = "", logpath: str = "./logs"):
    def console_filter(record):
        return not record["extra"].get("file_only", False)

    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add(sys.stdout, level="INFO", format=LOG_FORMAT, filter=console_filter)
    logger.add(f"{logpath}/log", rotation="500 MB", level="INFO", format=LOG_FORMAT)
    return logger.bind(prefix=prefix)


def run_resnet(
    input_loc,
    imagenet_label_dict,
    device,
    test_infra,
    model_version,
    delay_time_sec=0.1,
    n_times=1,
    use_trace=False,
    demo_vis=False,
    demo_fullscreen=False,
    demo_window_name="[BOS] ResNet50 Demo",
):
    logger = logger_setup()
    metric = evaluate.load("accuracy")
    ds = load_dataset(path="Tsomaros/Imagenet-1k_validation", cache_dir=CACHE_DIR, split="validation")
    prepared_ds = ds.with_transform(
        lambda batch: transform(batch, torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms())
    )
    dataloader = torch.utils.data.DataLoader(
        prepared_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True
    )
    first_batch_inputs = next(iter(dataloader))["pixel_values"]
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, first_batch_inputs)
    # Compile
    profiler.start("compile")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    if use_trace:
        spec = test_infra.input_tensor.spec
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    test_infra.output_tensor.deallocate(force=True)
    # Cache
    profiler.start("cache")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")

    # Trace capture
    if use_trace:
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        test_infra.output_tensor.deallocate(force=True)
        trace_input_addr = test_infra.input_tensor.buffer_address()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        tt_output_res = test_infra.run()
        tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        assert trace_input_addr == tt_image_res.buffer_address()
    else:
        test_infra.output_tensor.deallocate(force=True)

    ttnn.synchronize_device(device)
    CORRECT = 0
    TOTAL = 0
    for it, batch in enumerate(tqdm(dataloader)):
        inputs = batch["pixel_values"]
        reference = batch["labels"]
        tt_inputs_host_batch, _ = test_infra.setup_l1_sharded_input(device, inputs)
        profiler.start(f"inference_batch_{it}")
        if use_trace:
            ttnn.copy_host_to_device_tensor(tt_inputs_host_batch, tt_image_res)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        else:
            test_infra.input_tensor = tt_inputs_host_batch.to(device, input_mem_config)
            tt_output_res = test_infra.run()
        tt_out = ttnn.from_device(tt_output_res, blocking=True)
        profiler.end(f"inference_batch_{it}")
        tt_out_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).to(torch.float)
        prediction = tt_out_torch[:, 0, 0, :].argmax(dim=-1)
        metric.add_batch(predictions=prediction, references=reference)
        correct = (prediction == reference).sum().item()
        CORRECT += correct
        TOTAL += reference.numel()
    acc = metric.compute()["accuracy"]
    logger.info(f"model acc : {acc*100:.4f}%")
    if use_trace:
        ttnn.release_trace(device, tid)
    else:
        test_infra.output_tensor.deallocate(force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run for ResNet50")
    parser.add_argument("--trace", action="store_true", help="Use trace")
    parser.add_argument("--modelpath", type=str, help="path to resnet50 weight file")
    parser.add_argument("-data", type=str, default="sample", help="Choose in [sample, imagenet-val]")
    parser.add_argument("-delay", type=float, default=0.0, help="Delay time(sec), floating number")
    parser.add_argument("-n", type=int, default=1, help="Number of iteration")
    parser.add_argument("--demo", action="store_true", help="Show demo window")
    parser.add_argument("--fullscreen", action="store_true", help="Show demo window full size screen")
    args = parser.parse_args()

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=1605632)
    device.enable_program_cache()

    # input_loc = f"models/bos_model/ms_resnet/demo/images/{args.data}/"
    # imagenet_label_dict = ast.literal_eval(open("models/bos_model/ms_resnet/imagenet_class_labels.txt", "r").read())
    # model_version = "microsoft/resnet-50"
    input_loc = None
    imagenet_label_dict = None
    model_version = None

    test_infra = create_test_infra(
        device,
        BATCH_SIZE,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        args.modelpath,
        True,
        ttnn.L1_MEMORY_CONFIG,
        None,
    )
    run_resnet(
        input_loc,
        imagenet_label_dict,
        device,
        test_infra,
        model_version,
        args.delay,
        args.n,
        args.trace,
        args.demo,
        args.fullscreen,
    )
    ttnn.close_device(device)
