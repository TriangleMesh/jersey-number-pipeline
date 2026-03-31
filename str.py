#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass
from typing import List
from typing import Optional
import os
import copy

ROOT = './str/parseq/'
sys.path.append(str(ROOT))  # add ROOT to PATH


import torch
from torch import nn, optim
from torch.nn import functional as F

from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

from PIL import Image
import json
import csv


def build_tensorrt_model(model, img_size, args):
    if (not str(args.device).startswith('cuda')) or not torch.cuda.is_available():
        raise RuntimeError('TensorRT backend requires CUDA device. Use --device=cuda and a CUDA-capable GPU.')

    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError(
            'TensorRT backend requested but torch_tensorrt is not installed in the current environment.'
        ) from exc

    height, width = int(img_size[0]), int(img_size[1])
    min_batch = max(1, int(args.trt_min_batch))
    opt_batch = max(min_batch, int(args.trt_opt_batch))
    max_batch = max(opt_batch, int(args.trt_max_batch))
    compile_batch = opt_batch

    use_fp16 = args.trt_precision.lower() == 'fp16'
    enabled_precisions = {torch.float32, torch.float16} if use_fp16 else {torch.float32}

    class FixedLengthPARSeqWrapper(torch.nn.Module):
        def __init__(self, parseq_model, max_length):
            super().__init__()
            self.parseq_model = parseq_model
            self.max_length = int(max_length)

        def forward(self, images):
            # Fixed max_length disables PARSeq's data-dependent early-stop branch.
            return self.parseq_model(images, max_length=self.max_length)

    compile_max_length = max(1, int(args.trt_max_length))
    trt_base_model = copy.deepcopy(model)
    if hasattr(trt_base_model, 'decode_ar'):
        trt_base_model.decode_ar = False
    if hasattr(trt_base_model, 'refine_iters'):
        trt_base_model.refine_iters = 0
    trt_compile_model = FixedLengthPARSeqWrapper(trt_base_model, compile_max_length).eval().to('cuda')

    with torch.inference_mode():
        compiled = torch_tensorrt.compile(
            trt_compile_model,
            ir='dynamo',
            inputs=[
                torch_tensorrt.Input(
                    shape=(compile_batch, 3, height, width),
                    dtype=torch.float32,
                )
            ],
            enabled_precisions=enabled_precisions,
            workspace_size=int(args.trt_workspace_size_mb) * (1 << 20),
            truncate_long_and_double=True,
        )
    return compiled

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


def run_inference(model, data_root, result_file, img_size, trt_model: Optional[torch.nn.Module] = None):
    # load images one by one, save paths and result
    file_dir = os.path.join(data_root, 'imgs')
    filenames = os.listdir(file_dir)
    filenames.sort()
    results = {}
    for filename in tqdm(filenames):
        image = Image.open(os.path.join(file_dir, filename)).convert('RGB')
        transform = SceneTextDataModule.get_transform(img_size)
        image = transform(image)
        image = image.unsqueeze(0)
        device = model.device
        image = image.to(device)
        if trt_model is None:
            logits = model.forward(image)
        else:
            logits = trt_model(image)
        #convert to 3 by 10
        probs_full = logits[:,:3,:11].softmax(-1)
        preds, probs = model.tokenizer.decode(probs_full)
        logits = logits[:,:3,:11].cpu().detach().numpy()[0].tolist()
        # probs = logits.softmax(-1)
        # preds, probs = model.tokenizer.decode(probs)
        probs_full = probs_full.cpu().detach().numpy()[0].tolist()
        confidence = probs[0].cpu().detach().numpy().squeeze().tolist()
        results[filename] = {'label':preds[0], 'confidence':confidence, 'raw': probs_full, 'logits':logits}
    with open(result_file, 'w') as f:
        json.dump(results, f)


#================================ temperature scaling ======================================#


def run_inference_with_temperature(model, data_root, img_size):
    # load images one by one, save paths and result
    file_dir = os.path.join(data_root, 'imgs')
    filenames = os.listdir(file_dir)
    filenames.sort()
    results = {}
    for filename in filenames:
        image = Image.open(os.path.join(file_dir, filename)).convert('RGB')
        transform = SceneTextDataModule.get_transform(img_size)
        image = transform(image)
        image = image.unsqueeze(0)
        logits = model.forward(image.to(model.device))
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)
        confidence = probs[0].cpu().detach().numpy().squeeze().tolist()
        results[filename] = {'label':preds[0], 'confidence':confidence}


def temperature_scale(logits, t):
    # Expand temperature to match the size of logits
    temp = t.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2))
    new_logits = torch.div(logits, temp)
    return new_logits

temperature = nn.Parameter(torch.ones(1).cuda() * 1.5)
def set_temperature(model, data_root, img_size):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    model.cuda()
    ece_criterion = _ECELoss(model).cuda()

    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    file_dir = os.path.join(data_root, 'imgs')

    labels_file = os.path.join(data_root, 'test_gt.txt')
    reader = csv.reader(open(labels_file, 'r'))
    data = {}
    for row in reader:
        k, v = row
        data[k] = v

    myKeys = list(data.keys())
    with torch.no_grad():
        for filename in myKeys:
            image = Image.open(os.path.join(file_dir, filename)).convert('RGB')
            transform = SceneTextDataModule.get_transform(img_size)
            image = transform(image)
            image = image.unsqueeze(0)
            l = model.forward(image.to(model.device))
            logits_list.append(l)
            labels_list.append(data[filename])
        logits = torch.cat(logits_list).cuda()
        labels = labels_list

    # Calculate ECE before temperature scaling
    before_temperature_ece = ece_criterion(logits, labels).item()
    print('Before temperature -  ECE: %.3f' % (before_temperature_ece))

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = ece_criterion(temperature_scale(logits, temperature), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # Calculate NLL and ECE after temperature scaling
    after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After temperature - ECE: %.3f' % ( after_temperature_ece))

    return model


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, model, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.model = model

    def forward(self, logits, labels):
        targets = self.model.tokenizer.encode(labels, self.model.device)
        probs = logits.softmax(-1)

        confidences, temp_p = torch.max(probs[:,:3,:], dim=2)
        confidences = confidences.prod(1)

        temp_t = targets[:, 1:]
        accuracies = temp_p.eq(temp_t)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--std', action='store_true', default=False, help='Evaluate on standard benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom personal datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', default='torch', choices=['torch', 'tensorrt'],
                        help='Inference backend for STR model execution')
    parser.add_argument('--trt_precision', default='fp16', choices=['fp16', 'fp32'],
                        help='TensorRT precision mode when --backend=tensorrt')
    parser.add_argument('--trt_min_batch', type=int, default=1,
                        help='Minimum TensorRT optimization profile batch size')
    parser.add_argument('--trt_opt_batch', type=int, default=1,
                        help='Optimal TensorRT optimization profile batch size')
    parser.add_argument('--trt_max_batch', type=int, default=1,
                        help='Maximum TensorRT optimization profile batch size')
    parser.add_argument('--trt_workspace_size_mb', type=int, default=2048,
                        help='TensorRT workspace size in MB')
    parser.add_argument('--trt_max_length', type=int, default=2,
                        help='Fixed maximum decoded text length used for TensorRT compilation')
    parser.add_argument('--inference', action='store_true', default=False, help='Run inference and store prediction results')
    parser.add_argument('--tune_temperature', action='store_true', default=False,
                        help='Find best t-scale')
    parser.add_argument('--result_file', default='outputs/preds.json')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # PyTorch>=2.6 defaults to weights_only=True, which breaks older Lightning checkpoints.
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

    charset_test = string.digits # + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase
    if args.punctuation:
        charset_test += string.punctuation
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams

    trt_model = None
    if args.backend == 'tensorrt':
        print('Compiling PARSeq model with TensorRT...')
        trt_model = build_tensorrt_model(model, hp.img_size, args)
        print('TensorRT compilation finished.')

    if args.inference:
        run_inference(model, args.data_root, args.result_file, hp.img_size, trt_model=trt_model)
        exit()
    if args.tune_temperature:
        set_temperature(model, args.data_root, hp.img_size)
        exit()


    datamodule = SceneTextDataModule(args.data_root, '_unused_', hp.img_size, 2, hp.charset_train,
                                     hp.charset_test, args.batch_size, args.num_workers, False, rotation=args.rotation)



    test_set = ['JerseyNumbers']

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            print(res)
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)
        print(f"accuracy:{accuracy}, mean_conf:{mean_conf}")

    result_groups = {
        'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB,
        'Benchmark': SceneTextDataModule.TEST_BENCHMARK
    }
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                #print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()