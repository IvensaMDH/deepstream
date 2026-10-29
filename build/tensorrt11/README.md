# Manual TensorRT 11 inference build

Run **Actions → Build TensorRT 11 inference libraries → Run workflow**.
The workflow builds only `libnvds_infer.so`, `libnvds_inferutils.so`, and
`libnvds_inferlogger.so`. It does not build DeepStream, change the GStreamer
`nvinfer` plugin, create releases, or update/rebase the fork.

Download the `nvinfer-trt11-linux-x86_64` Actions artifact. It contains
`nvinfer-trt11-linux-x86_64.tar.gz`, with the three libraries, checksums, license
notices, and `BUILDINFO` identifying the checked-out source and toolkit versions.

```bash
gh run download RUN_ID --repo IvensaMDH/deepstream \
  --name nvinfer-trt11-linux-x86_64 --dir ./trt11-artifact
tar -xzf trt11-artifact/nvinfer-trt11-linux-x86_64.tar.gz
(cd nvinfer-trt11 && sha256sum -c SHA256SUMS)
```

The build uses Ubuntu 24.04, CUDA 13.2, and TensorRT 11.4.0.106. Version pins are
in `.github/workflows/build-trt11.yml`. CUDA kernels include native targets for
T4 (75), RTX/Ampere (86), and L4/L40S (89). After a manual rebase, rerun the workflow;
update the toolkit pins yourself if the new source needs different versions.
The artifact's DeepStream version comes from the checked-out Makefile. Use it
only with that matching SDK version, not with older DeepStream plugin binaries.

The TensorRT package is NVIDIA's CUDA-13.x build (its package suffix is
`cuda13.4`); the companion kernels are compiled with 13.2 to align with the
deployment fleet. This does not waive driver requirements or guarantee every
TensorRT11 model works on an older driver. Validate on the actual deployment GPU.

## Add to a Docker build

Upgrade the SDK package, headers and plugin paths to DeepStream 9.2, and install
matching TensorRT 11 runtime libraries. Keep the original `nvinfer` plugin.
Copy the downloaded archive into the Docker build context. In the inference
stage, **before collecting runtime dependencies with `ldd`**:

```dockerfile
COPY trt11-artifact/nvinfer-trt11-linux-x86_64.tar.gz /tmp/
RUN tar -xzf /tmp/nvinfer-trt11-linux-x86_64.tar.gz -C /tmp && \
    cd /tmp/nvinfer-trt11 && sha256sum -c SHA256SUMS && \
    cp lib/*.so /opt/nvidia/deepstream/deepstream-9.2/lib/
```

For a trimmed runtime image, collect dependencies after this overlay so the
collector copies TensorRT 11 and the new companions, not the old libraries.
If you retain `trtexec`, collect its dependencies too: the engine-only inference
library no longer imports the ONNX parser, but `trtexec` still needs it. Keep its
matching builder resources for the GPU architectures on which engines are built.
Do not copy build credentials or GitHub tokens into image layers.

This workflow verifies compilation and library resolution, **not GPU inference**.
Test the finished image and your real model on the deployment GPU before use.
The port requires externally built, matching TensorRT 11 engines; automatic
ONNX/TLT conversion and legacy calibration are disabled. Tracker, Triton and
other TensorRT 10 users are not covered. Do not mix their incompatible interfaces
into the same process.

To build locally with an installed development stack:

```bash
CUDA_VER=13.2 bash build/tensorrt11/build.sh /tmp/trt11-output
```
