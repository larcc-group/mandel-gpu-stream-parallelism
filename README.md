
# How to Cite This Work

Rockenbach, D. A.; Stein, C. M.; Griebler, D.; Mencagli, G.; Torquati, M.; Danelutto, M.; Fernandes, L. G. **Stream Processing on Multi-Cores with GPUs: Parallel Programming Models' Challenges**. *IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*. Rio de Janeiro, Brazil, 2019.

```bibtex
@inproceedings{larcc:stream_processing_gpu_challenges:IPDPSW:19,
    author={Dinei A. Rockenbach and Charles M. Stein and Dalvan Griebler and Gabriele Mencagli and Massimo Torquati and Marco Danelutto and Luiz Gustavo Fernandes},
    title={{Stream Processing on Multi-Cores with GPUs: Parallel Programming Models' Challenges}},
    booktitle={IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
    pages={834-841},
    year={2019},
    address={Rio de Janeiro, Brazil},
    month={May},
    publisher={IEEE},
    doi={10.1109/IPDPSW.2019.00137},
    url={https://ieeexplore.ieee.org/document/8778359},
}
```

# mandel-gpu-stream-parallelism

This repository has the parallel and sequential implementations for the Mandelbrot Streaming pseudo application. We introduced stream parallelism for CPU (using SPar, TBB and FastFlow) and GPU (using CUDA and OpenCL).

## Compiling

To compile all the project source files, you need to have the following prerequisites:

* SPar in your home folder (https://github.com/dalvangriebler/SPar)
  * SPar automatically download the FastFlow library when compiled
* Threading Building Blocks (https://www.threadingbuildingblocks.org/)
* NVIDIA CUDA
  * The NVIDIA CUDA Toolkit has the OpenCL libraries needed for the OpenCL versions

Once you have all the prerequisites installed, enter the `src` folder and run the `make` command.

This will generate various runnable files prefixed with `bin_`
