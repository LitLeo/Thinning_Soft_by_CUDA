#Thinning#

---
在“Thinning”文件夹中是六个分别根据不同论文实现的细化算法。每个算法分别基于 **CUDA** 和 **OpenCV** 实现。OpenCV是正常串行版本，CUDA是并行迭代版本。以下是六篇论文：

1. DP.(论文找不到了=.=)

2. "Guo, Hall - 1989 - Parallel thinning with two-subiteration algorithms".(GH)

3. "Han, La, Rhee - 1997 - An efficient fully parallel thinning algorithm".(Han)

4. "Petrosino, Salvi - 2000 - A two-subcycle thinning algorithm and its parallel implementation on SIMD machines".(PS)

5. "A fast parallel algorithm for thinning digital patterns" by T.Y. Zhang and C.Y. Suen.(ZS)

6. "Kwon, Gi, Kang - Unknown - An enhanced thinning algorithm using parallel processing".(Kwon)

分别使用不同的算法对输入的八位灰度图像进行细化处理。其中*Kwon*是对*ZS*的改进算法。

可以使用每个算法相应的接口函数(thinDP(),thinGH(),thinHan(),thinPet()，thinZS(),thinKwon())，输出对应算法处理后的细化图像。

---

在“Thinning_Imp”文件夹中是分别基于GH，PS，ZS算法，运用 **滑动窗口策略** 而实现的并行优化算法。通过调换核函数的参数，省去数据的重复拷贝的过程。此外，调换参数将已经处理过的图像放在参考图像的位置，通过相同位置点的比较，可以省去重复的判断。节省资源，进行加速优化。以GH算法为例：

(原算法)

		cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);

        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }
         _thinGpuIter1Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount,
                                                     highPixel, lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }

        cudaerrcode = cudaMemcpyPeer(tempimg->imgData, tempsubimgCud.deviceId, 
                                  outimg->imgData, outsubimgCud.deviceId, 
                                  outsubimgCud.pitchBytes * outimg->height);

        if (cudaerrcode != cudaSuccess) {
            return CUDA_ERROR;
        }

        _thinGpuIter2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud,
                                                     devchangecount, highPixel, lowPixel);
        if (cudaGetLastError() != cudaSuccess) {
            return CUDA_ERROR;
        }     

        

(优化算法)
	
		/*如果已经进行了奇数次迭代，那么现在源图像是在outimg上，输出图像应该在tempimg上*/
		_thinGpuPtIter1Ker<<<gridsize, blocksize>>>(outsubimgCud, tempsubimgCud, devchangecount,
	                                                    highPixel, lowPixel);
        if (cudaGetLastError() = cudaSuccess) {
            i++;//根据变量i来判断迭代次数
        }

        else
            return CUDA_ERROR;
            
        _thinGpuPtIter2Ker<<<gridsize, blocksize>>>(tempsubimgCud, outsubimgCud, devchangecount, 
                                                    highPixel, lowPixel);
        if (cudaGetLastError() = cudaSuccess) {
            i++;
        }

        else
            return CUDA_ERROR;


---

以下为不同策略的算法时间对比(100次)

	|    |   OpenCV   |   cuda    |  cuda_Pt  |
	|----|:----------:|:---------:|:---------:|
	| GH |  151762ms  | 1214.72ms | 1056.37ms |
	| PS |   31498ms  | 944.281ms | 868.735ms |
	| ZS |   93214ms  | 895.417ms | 803.763ms |

