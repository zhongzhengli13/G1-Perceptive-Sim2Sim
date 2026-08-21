import os
import numpy as np
import onnxruntime as ort
from typing import Dict, Union

class OnnxPolicyInference:
    _device_dict = {"cpu": "CPUExecutionProvider", "cuda:0": "CUDAExecutionProvider"}

    _dtype_dict = {
        "tensor(float)": np.float32,
    }

    def __init__(self, onnx_file: str, device: str = "cpu"):
        device = device.lower()
        if (
            device not in self._device_dict
            or self._device_dict[device] not in ort.get_available_providers()
        ):
            print(f"Unsupported device type {device}, fall back to cpu")
            device = "cpu"
        if not os.path.exists(onnx_file):
            raise FileNotFoundError(f"No such file called {onnx_file}")
        self.onnx_file = onnx_file
        self.provider = self._device_dict[device]
        self.session = ort.InferenceSession(onnx_file, providers=[self.provider])
        self._init_io_infos()

    def __str__(self):
        return (
            f"{self.onnx_file}: "
            f"{len(self.input_infos)} inputs: {[(k, v[0], v[1]) for k, v in self.input_infos.items()]}, "
            f"{len(self.output_infos)} outputs: {[(k, v[0], v[1]) for k, v in self.output_infos.items()]}"
        )

    def _init_io_infos(self):
        self.input_infos = {}
        self.output_infos = {}
        for input_ in self.session.get_inputs():
            self.input_infos[input_.name] = (
                input_.shape,
                self._dtype_dict[input_.type],
            )

        for output_ in self.session.get_outputs():
            self.output_infos[output_.name] = (
                output_.shape,
                self._dtype_dict[output_.type],
            )

    def _check_input_feed(self, input_feed: Dict[str, np.ndarray]):
        for k, v in input_feed.items():
            if k not in self.input_infos:
                raise ValueError(f"No such input called {k}")
            if v.shape != tuple(self.input_infos[k][0]):
                raise ValueError(
                    f"shape mismatch required {tuple(self.input_infos[k][0])} while get {v.shape}"
                )
            if v.dtype != self.input_infos[k][1]:
                raise ValueError(
                    f"dtype mismatch required {self.input_infos[k][1]} while get {v.dtype}"
                )

    def forward(
        self, input_feed: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        if isinstance(input_feed, np.ndarray):
            input_feed = {"obs": input_feed}
        self._check_input_feed(input_feed)
        return self.session.run(
            output_names=list(self.output_infos.keys()), input_feed=input_feed
        )

    def __call__(
        self, input_feed: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        return self.forward(input_feed)[0]
